"""Text-preserving model of the editable numbers in a TDB file.

The file text is never re-generated: every editable number is located by its character span, and saving
replaces only the spans whose value changed, so the rest of the file stays byte-identical.

Two kinds of editable rows:

* **VV rows**: a term of a PARAMETER (or FUNCTION) expression that is one ESPEI symbol (``VV0012``) times a
  temperature basis (``1, T, T*LN(T), T**2, T**3, T**(-1)``). The value lives in ``FUNCTION VV0012 ...``. A
  symbol used in several terms gives one row per use; they all show and edit the same value.
* **Coefficient rows**: a term that is a plain number times a temperature basis (``-24.112*T*LN(T)``).

Coefficient letters follow ``a + b*T + c*T*ln(T) + d*T**2 + e*T**3 + f*T**(-1)``.

Some keyword handling and the comment masking are adapted from calphad-agent's ``tdbparse``/``param_tools``.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

KEYWORDS = [
    "ELEMENT", "SPECIES", "PHASE", "PARAMETER", "CONSTITUENT", "FUNCTION",
    "TYPE_DEFINITION", "DEFINE_SYSTEM_DEFAULT", "DEFAULT_COMMAND",
    "DATABASE_INFO", "VERSION_DATE", "VERSION_DATA", "TEMPERATURE_LIMITS",
    "LIST_OF_REFERENCES", "ADD_REFERENCES", "REFERENCE_FILE", "ASSESSED_SYSTEMS",
    "DIFFUSION", "ZERO_VOLUME_SPECIES",
]

VV_RE = re.compile(r"^VV\d+$")
# Prefix of the hidden symbols that stand for edited coefficients in the compute database.
LIVE_PREFIX = "TWKC"

BASIS_ORDER = ["const", "T", "TLNT", "T2", "T3", "TINV"]
BASIS_LETTER = {"const": "a", "T": "b", "TLNT": "c", "T2": "d", "T3": "e", "TINV": "f"}
BASIS_TEXT = {"const": "", "T": "T", "TLNT": "T*LN(T)", "T2": "T**2", "T3": "T**3", "TINV": "T**(-1)",
              "LNT": "LN(T)"}


def resolve_keyword(word: str) -> Optional[str]:
    w = word.upper().replace("-", "_")
    if not w:
        return None
    if w in KEYWORDS:
        return w
    wparts = w.split("_")
    for kw in KEYWORDS:
        kparts = kw.split("_")
        if len(wparts) <= len(kparts) and all(k.startswith(p) for p, k in zip(wparts, kparts)):
            if len(wparts) == 1 and len(w) < 3:
                continue
            return kw
    return None


def mask_comments(text: str) -> str:
    """Same length as ``text`` with ``$`` comments blanked out."""
    out = []
    for ln in text.splitlines(keepends=True):
        body = ln.rstrip("\r\n")
        nl = ln[len(body):]
        if body.strip().startswith("$"):
            out.append(" " * len(body) + nl)
            continue
        if "$" in body and "'" not in body.split("$", 1)[0]:
            i = body.index("$")
            body = body[:i] + " " * (len(body) - i)
        out.append(body + nl)
    return "".join(out)


def fmt_number(x: float) -> str:
    """Shortest text that reads back as exactly ``x`` (TDB style exponent)."""
    if x == 0:
        return "0"
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    s = repr(float(x)).upper()
    if "E" in s:
        m, e = s.split("E")
        if "." not in m:
            m += ".0"
        s = f"{m}E{int(e):+d}" if not e.startswith(("+", "-")) else f"{m}E{e}"
    return s


def basis_label(basis: str) -> str:
    if basis in BASIS_LETTER:
        return BASIS_LETTER[basis]
    if basis == "LNT":
        return "ln(T)"
    if basis.startswith("T"):
        return f"T^{basis[1:]}"
    return basis


def basis_rank(basis: str) -> Tuple[int, float]:
    if basis in BASIS_ORDER:
        return (BASIS_ORDER.index(basis), 0.0)
    if basis == "LNT":
        return (len(BASIS_ORDER), 0.0)
    try:
        return (len(BASIS_ORDER) + 1, float(basis[1:]))
    except ValueError:
        return (len(BASIS_ORDER) + 2, 0.0)


def basis_from_exponents(texp: float, lnt: int) -> Optional[str]:
    if lnt == 0:
        if texp == 0:
            return "const"
        if texp == 1:
            return "T"
        if texp == -1:
            return "TINV"
        if texp == int(texp):
            return f"T{int(texp)}"
        return None
    if lnt == 1 and texp == 1:
        return "TLNT"
    if lnt == 1 and texp == 0:
        return "LNT"
    return None


# ---------------------------------------------------------------------------------------------------------
# expressions
# ---------------------------------------------------------------------------------------------------------
_TOKEN_RE = re.compile(
    r"\s*(?:(?P<num>(?:\d+\.?\d*|\.\d+)(?:[EeDd][+-]?\d+)?)"
    r"|(?P<name>[A-Za-z_][A-Za-z0-9_]*#?)"
    r"|(?P<op>\*\*|[-+*/(),^]))"
)


def _tokens(text: str, offset: int) -> List[Tuple[str, str, int, int]]:
    """(kind, value, abs_start, abs_end); raises ValueError on characters it does not know."""
    out, pos = [], 0
    while pos < len(text):
        if text[pos:].strip() == "":
            break
        m = _TOKEN_RE.match(text, pos)
        if not m or m.end() == pos:
            raise ValueError(f"cannot read expression near {text[pos:pos + 20]!r}")
        kind = m.lastgroup
        val = m.group(kind)
        out.append((kind, val, offset + m.start(kind), offset + m.end(kind)))
        pos = m.end()
    return out


@dataclass
class Term:
    start: int                 # absolute span in the file, including the leading +/- operator if any
    end: int
    has_op: bool               # True when the term starts with its own +/- operator
    sign: int
    factors: List[str]         # factor texts as written (top level '*'), uppercase
    num: float                 # product of the numeric factors (1.0 when there are none)
    n_num: int                 # number of numeric factors
    symbols: List[str]         # non-T names (without '#')
    basis: Optional[str]       # const/T/TLNT/T2/T3/TINV/LNT/T<n>, None when not a monomial in T

    @property
    def nonnum_factors(self) -> List[str]:
        return [f for f in self.factors if not _is_number(f)]


def _is_number(s: str) -> bool:
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    try:
        float(s.replace("D", "E").replace("d", "e"))
        return True
    except ValueError:
        return False


def _num(s: str) -> float:
    s = s.strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    return float(s.replace("D", "E").replace("d", "e"))


_POW_RE = re.compile(r"^T(?:\*\*|\^)\(?([+-]?\d+(?:\.\d+)?)\)?$")


def _classify_factors(factors: List[str], divisors: List[str]):
    num, n_num, texp, lnt, symbols, ok = 1.0, 0, 0.0, 0, [], True
    for f, power in [(f, 1) for f in factors] + [(d, -1) for d in divisors]:
        if _is_number(f):
            v = _num(f)
            num *= v if power == 1 else 1.0 / v
            n_num += 1
            continue
        if f == "T":
            texp += power
            continue
        if f in ("LN(T)", "LOG(T)"):
            if power != 1:
                ok = False
            lnt += 1
            continue
        m = _POW_RE.match(f)
        if m:
            texp += power * float(m.group(1))
            continue
        if re.fullmatch(r"[A-Z_][A-Z0-9_]*#?", f):
            if power != 1:
                ok = False
            symbols.append(f.rstrip("#"))
            continue
        ok = False
    basis = basis_from_exponents(texp, lnt) if ok else None
    return num, n_num, symbols, basis


def split_terms(text: str, offset: int = 0) -> List[Term]:
    """Split an expression (file text ``text`` starting at absolute ``offset``) into top-level terms."""
    toks = _tokens(text, offset)
    groups: List[Tuple[int, List]] = []    # (sign, tokens)
    cur: List = []
    sign, depth, last_value = 1, 0, False
    starts: List[Tuple[int, bool]] = []
    term_start, has_op = None, False
    for tok in toks:
        kind, val, s, e = tok
        if kind == "op" and val in "+-" and depth == 0 and (not cur or last_value):
            if cur:
                groups.append((sign, cur))
                starts.append((term_start, has_op))
                cur, sign = [], 1
                term_start, has_op = None, False
            if term_start is None:
                term_start, has_op = s, True
            sign *= -1 if val == "-" else 1
            last_value = False
            continue
        if term_start is None:
            term_start, has_op = s, False
        if val == "(":
            depth += 1
        elif val == ")":
            depth -= 1
        cur.append(tok)
        last_value = kind in ("num", "name") or val == ")"
    if cur:
        groups.append((sign, cur))
        starts.append((term_start, has_op))
    terms = []
    for (sign, toks_), (ts, hop) in zip(groups, starts):
        # split at top-level '*' and '/'
        factors, divisors, buf, depth, mode = [], [], [], 0, "*"
        for kind, val, s, e in toks_:
            if val == "(":
                depth += 1
            elif val == ")":
                depth -= 1
            if depth == 0 and val in ("*", "/") and kind == "op":
                (factors if mode == "*" else divisors).append("".join(buf).upper())
                buf, mode = [], val
                continue
            buf.append(val)
        if buf:
            (factors if mode == "*" else divisors).append("".join(buf).upper())
        num, n_num, symbols, basis = _classify_factors(factors, divisors)
        if divisors:
            # rebuilt text would be ambiguous; only allow plain classification, never rewriting numbers
            n_num = -1 if n_num else n_num
        terms.append(Term(ts, toks_[-1][3], hop, sign, factors + [f"/{d}" for d in divisors], num, n_num,
                          symbols, basis))
    return terms


# ---------------------------------------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------------------------------------
@dataclass
class Range:
    tlow: Optional[float]
    thigh: Optional[float]
    expr_start: int
    expr_end: int
    terms: List[Term]


@dataclass
class Command:
    kind: str                  # PARAMETER / FUNCTION / ELEMENT / SPECIES / ...
    start: int
    end: int
    text: str
    name: str = ""
    ranges: List[Range] = field(default_factory=list)
    error: str = ""


_NUMTOK = re.compile(r"\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][+-]?\d+)?)(?=[\s;,]|$)")


def parse_piecewise(masked: str, start: int, end: int) -> List[Range]:
    """Parse ``[Tlow] expr; [Thigh] Y expr; [Thigh] N [ref]`` in masked[start:end]."""
    ranges = []
    pos = start
    m = _NUMTOK.match(masked, pos, end)
    tlow = None
    if m:
        tlow = float(m.group(1).replace("D", "E").replace("d", "e"))
        pos = m.end()
    while True:
        semi = masked.find(";", pos, end)
        if semi < 0:
            raise ValueError("expression without ';'")
        seg = masked[pos:semi]
        lead = len(seg) - len(seg.lstrip())
        es, ee = pos + lead, pos + len(seg.rstrip())
        terms = split_terms(masked[es:ee], es) if ee > es else []
        pos = semi + 1
        m = _NUMTOK.match(masked, pos, end)
        thigh = None
        if m:
            thigh = float(m.group(1).replace("D", "E").replace("d", "e"))
            pos = m.end()
        ranges.append(Range(tlow, thigh, es, ee, terms))
        flag = re.compile(r"\s*([YyNn])").match(masked, pos, end)
        if flag and flag.group(1) in "Yy":
            pos = flag.end()
            tlow = thigh
            continue
        break
    return ranges


def scan_commands(text: str) -> List[Command]:
    masked = mask_comments(text)
    cmds, prev = [], 0
    for e in [i for i, ch in enumerate(masked) if ch == "!"]:
        seg = masked[prev:e]
        if seg.strip():
            start = prev + (len(seg) - len(seg.lstrip()))
            body = masked[start:e]
            words = body.split()
            kw = resolve_keyword(words[0]) or words[0].upper()
            c = Command(kw, start, e + 1, text[start:e + 1])
            try:
                if kw == "PARAMETER":
                    after_kw = start + len(body) - len(body.split(None, 1)[1])
                    close = masked.index(")", after_kw, e)
                    c.name = re.sub(r"\s+", "", masked[after_kw:close + 1]).upper()
                    c.ranges = parse_piecewise(masked, close + 1, e)
                elif kw == "FUNCTION":
                    c.name = words[1].upper().rstrip("#")
                    name_end = masked.index(words[1], start + len(words[0])) + len(words[1])
                    c.ranges = parse_piecewise(masked, name_end, e)
                elif kw in ("ELEMENT", "SPECIES", "PHASE", "CONSTITUENT"):
                    c.name = words[1].upper() if len(words) > 1 else ""
            except (ValueError, IndexError) as ex:
                c.error = str(ex)
            cmds.append(c)
        prev = e + 1
    return cmds


# ---------------------------------------------------------------------------------------------------------
# parameter names
# ---------------------------------------------------------------------------------------------------------
@dataclass
class ParamName:
    ptype: str                                  # G, L, TC, BMAGN, ...
    phase: str
    sublattices: Tuple[Tuple[str, ...], ...]   # as written
    order: int

    @property
    def interacting(self) -> bool:
        return any(len(sl) > 1 for sl in self.sublattices)

    @property
    def kind(self) -> str:
        """G (end member), L (interaction) or the property name (TC, BMAGN, ...)."""
        if self.ptype in ("G", "L"):
            return "L" if self.interacting else "G"
        return self.ptype

    @property
    def constituent_text(self) -> str:
        return ":".join(",".join(sl) for sl in self.sublattices)

    def term_label(self) -> str:
        k = self.kind
        if k == "L":
            return f"L{self.order}"
        if k == "G":
            return "G" if self.order == 0 else f"G{self.order}"
        return f"{k}{self.order}" if (self.interacting or self.order) else k


def parse_param_name(raw: str) -> ParamName:
    m = re.match(r"^\s*([A-Za-z0-9_]+)\s*\((.*)\)\s*$", raw, re.S)
    if not m:
        raise ValueError(f"bad parameter name {raw!r}")
    ptype = m.group(1).upper()
    ptype = {"BM": "BMAGN", "BMAG": "BMAGN"}.get(ptype, ptype)
    inner = re.sub(r"\s+", "", m.group(2)).upper()
    order = 0
    if ";" in inner:
        inner, o = inner.rsplit(";", 1)
        try:
            order = int(o) if o else 0
        except ValueError:
            order = 0
    phase, _, rest = inner.partition(",")
    phase = phase.split(":")[0].split("&")[0]
    subls = tuple(tuple(s.strip("%#") for s in sl.split(",") if s) for sl in rest.split(":"))
    return ParamName(ptype, phase, subls, order)


def kind_rank(pn: ParamName) -> Tuple[int, int, str]:
    """G, then L0, L1, L2, ..., then TC, BMAGN, then anything else."""
    k = pn.kind
    if k == "G":
        return (0, pn.order, "")
    if k == "L":
        return (1, pn.order, "")
    if k == "TC":
        return (2, pn.order, "")
    if k == "BMAGN":
        return (3, pn.order, "")
    return (4, pn.order, k)


# ---------------------------------------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------------------------------------
@dataclass
class Row:
    rid: int
    kind: str                       # "VV" or "COEF"
    symbol: str                     # VV name, or hidden live symbol name for COEF rows
    owner: str                      # "PARAMETER" or "FUNCTION"
    owner_name: str                 # raw parameter name or FUNCTION name
    pname: Optional[ParamName]
    range_index: int
    n_ranges: int
    trange: Tuple[Optional[float], Optional[float]]
    basis: str
    multiplier: float               # VV rows: the term is multiplier * VV * basis
    elements: FrozenSet[str]        # elements of the constituents (VA excluded); empty for FUNCTION rows
    term: Term
    command_index: int

    @property
    def phase(self) -> str:
        return self.pname.phase if self.pname else ""

    @property
    def letter(self) -> str:
        return basis_label(self.basis)

    def meaning(self) -> str:
        if self.pname is None:
            s = f"FUNCTION {self.owner_name}, {self.letter}"
        else:
            s = f"{self.pname.phase}, {self.pname.term_label()}({self.pname.constituent_text}), {self.letter}"
        if self.n_ranges > 1:
            lo, hi = self.trange
            s += f"  [{_tfmt(lo)}–{_tfmt(hi)} K]"
        if self.kind == "VV" and self.multiplier != 1.0:
            s += f"  (×{self.multiplier:g})"
        return s

    def phase_sort_key(self, by_constituents: bool = False):
        pn = self.pname
        if pn is None:
            base = ("~", (9, 0, ""), "", self.owner_name)
        else:
            base = (pn.phase, kind_rank(pn), pn.constituent_text, "")
            if by_constituents:
                base = (pn.phase, (len(self.elements), 0, ""), pn.constituent_text, kind_rank(pn))
        return base + (self.range_index, basis_rank(self.basis), self.symbol)

    def vv_sort_key(self):
        if self.kind == "VV":
            return (0, int(self.symbol[2:]), self.phase_sort_key())
        return (1, 0, self.phase_sort_key())


def _tfmt(t: Optional[float]) -> str:
    return "?" if t is None else f"{t:g}"


@dataclass
class VVDef:
    name: str
    value: float
    command_index: int
    expr_start: int
    expr_end: int


class TDBDocument:
    """A TDB file and the current values of its editable numbers."""

    def __init__(self, text: str, path: Optional[str] = None):
        self.path = path
        self.text = text
        self.commands = scan_commands(text)
        self.elements: Dict[str, float] = {}           # element -> mass
        self.species: Dict[str, FrozenSet[str]] = {}
        self.vvdefs: Dict[str, VVDef] = {}
        self.rows: List[Row] = []
        self.warnings: List[str] = []
        self._parse()
        self.vv_values: Dict[str, float] = {k: d.value for k, d in self.vvdefs.items()}
        self.vv_orig = dict(self.vv_values)
        self.coef_values: Dict[int, float] = {r.rid: self._coef_value(r) for r in self.rows if r.kind == "COEF"}
        self.coef_orig = dict(self.coef_values)
        self.live: set = set()                          # COEF rids that are symbolic in the compute database
        self.locked: set = set()                        # symbols (VV names, coefficient symbols) that must not change

    # -- loading ------------------------------------------------------------------------------------------
    @classmethod
    def from_file(cls, path: str) -> "TDBDocument":
        path = os.path.abspath(path)
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            return cls(f.read(), path)

    def _species_elements(self, name: str) -> FrozenSet[str]:
        name = name.upper().strip("%#")
        if name in ("VA", "/-"):
            return frozenset()
        if name in self.species:
            return self.species[name]
        if name in self.elements:
            return frozenset([name])
        return frozenset([name])

    def _formula_elements(self, formula: str) -> FrozenSet[str]:
        formula = formula.upper().split("/")[0]
        els, pos, names = set(), 0, sorted(self.elements, key=len, reverse=True)
        while pos < len(formula):
            for el in names:
                if formula.startswith(el, pos):
                    if el != "VA":
                        els.add(el)
                    pos += len(el)
                    break
            else:
                pos += 1
        return frozenset(els)

    def _parse(self):
        for c in self.commands:
            if c.kind == "ELEMENT":
                w = c.text.split()
                try:
                    self.elements[w[1].upper()] = float(w[3].replace("D", "E"))
                except (IndexError, ValueError):
                    if len(w) > 1:
                        self.elements[w[1].upper()] = float("nan")
        for c in self.commands:
            if c.kind == "SPECIES":
                w = c.text.rstrip("!").split()
                if len(w) >= 3:
                    self.species[w[1].upper()] = self._formula_elements(w[2])
        for ci, c in enumerate(self.commands):
            if c.error:
                if c.kind in ("PARAMETER", "FUNCTION"):
                    self.warnings.append(f"could not read {c.kind} {c.name or c.text[:40]!r}: {c.error}")
                continue
            if c.kind == "FUNCTION" and VV_RE.match(c.name):
                r = c.ranges
                if len(r) == 1 and len(r[0].terms) == 1 and r[0].terms[0].basis == "const" \
                        and not r[0].terms[0].symbols and r[0].terms[0].n_num >= 0:
                    t = r[0].terms[0]
                    self.vvdefs[c.name] = VVDef(c.name, t.sign * t.num, ci, r[0].expr_start, r[0].expr_end)
                else:
                    self.warnings.append(f"FUNCTION {c.name} is not a single constant; it is not editable")
        for ci, c in enumerate(self.commands):
            if c.error or c.kind not in ("PARAMETER", "FUNCTION"):
                continue
            if c.kind == "FUNCTION" and VV_RE.match(c.name):
                continue
            pname, els = None, frozenset()
            if c.kind == "PARAMETER":
                try:
                    pname = parse_param_name(c.name)
                except ValueError as ex:
                    self.warnings.append(str(ex))
                    continue
                els = frozenset().union(*[self._species_elements(s) for sl in pname.sublattices for s in sl]) \
                    if pname.sublattices else frozenset()
            for ri, rng in enumerate(c.ranges):
                for t in rng.terms:
                    if t.basis is None:
                        continue
                    if len(t.symbols) == 1 and t.symbols[0] in self.vvdefs:
                        kind, sym, mult = "VV", t.symbols[0], t.sign * t.num
                    elif not t.symbols and t.n_num == 1 and c.kind == "PARAMETER":
                        kind, sym, mult = "COEF", "", 1.0
                    else:
                        continue
                    rid = len(self.rows)
                    if kind == "COEF":
                        sym = f"{LIVE_PREFIX}{rid:05d}"
                    self.rows.append(Row(rid, kind, sym, c.kind, c.name, pname, ri, len(c.ranges),
                                         (rng.tlow, rng.thigh), t.basis, mult, els, t, ci))

    @staticmethod
    def _coef_value(r: Row) -> float:
        return r.term.sign * r.term.num

    # -- values -------------------------------------------------------------------------------------------
    def value(self, row: Row) -> float:
        return self.vv_values[row.symbol] if row.kind == "VV" else self.coef_values[row.rid]

    def original_value(self, row: Row) -> float:
        return self.vv_orig[row.symbol] if row.kind == "VV" else self.coef_orig[row.rid]

    def set_value(self, row: Row, value: float) -> bool:
        """Set a value. Returns True when the compute database has to be rebuilt (a new live coefficient)."""
        value = float(value)
        if row.kind == "VV":
            self.vv_values[row.symbol] = value
            return False
        self.coef_values[row.rid] = value
        if row.rid not in self.live:
            self.live.add(row.rid)
            return True
        return False

    def is_locked(self, row: Row) -> bool:
        return row.symbol in self.locked

    def set_locked(self, rows: Iterable[Row], on: bool) -> None:
        for r in rows:
            (self.locked.add if on else self.locked.discard)(r.symbol)

    def rows_of_phase(self, phase: str) -> List[Row]:
        return [r for r in self.rows if (r.phase or "(functions)") == phase]

    def is_modified(self, row: Row) -> bool:
        return self.value(row) != self.original_value(row)

    def modified_count(self) -> int:
        return sum(1 for k, v in self.vv_values.items() if v != self.vv_orig[k]) + \
            sum(1 for k, v in self.coef_values.items() if v != self.coef_orig[k])

    def parameter_values(self) -> Dict[str, float]:
        """Symbol -> value for every symbol that is a pycalphad parameter in the compute database."""
        vals = dict(self.vv_values)
        for rid in sorted(self.live):
            vals[self.rows[rid].symbol] = self.coef_values[rid]
        return vals

    # -- subsystems -----------------------------------------------------------------------------------------
    def element_list(self) -> List[str]:
        return sorted(e for e in self.elements if e not in ("VA", "/-"))

    def rows_in(self, elements: Iterable[str]) -> List[Row]:
        sel = frozenset(e.upper() for e in elements)
        return [r for r in self.rows if r.pname is None or r.elements <= sel]

    @staticmethod
    def is_lower_order(row: Row, elements: Sequence[str]) -> bool:
        return row.pname is not None and len(row.elements) < len(set(elements))

    # -- text -----------------------------------------------------------------------------------------------
    def _term_text(self, row: Row, value: Optional[float] = None, symbol: Optional[str] = None) -> str:
        t = row.term
        rest = [f for f in t.nonnum_factors]
        if symbol is not None:
            body = "*".join([symbol + "#"] + rest)
            return ("+" if t.has_op else "") + body
        v = float(value)
        mag = fmt_number(abs(v))
        body = "*".join([mag] + rest)
        if v < 0:
            return "-" + body
        return ("+" if t.has_op else "") + body

    def _apply(self, edits: List[Tuple[int, int, str]], text: Optional[str] = None) -> str:
        text = self.text if text is None else text
        out, last = [], 0
        for s, e, new in sorted(edits):
            out.append(text[last:s])
            out.append(new)
            last = e
        out.append(text[last:])
        return "".join(out)

    def render(self) -> str:
        """The file text with every changed value written in place."""
        edits = []
        for name, d in self.vvdefs.items():
            if self.vv_values[name] != self.vv_orig[name]:
                edits.append((d.expr_start, d.expr_end, fmt_number(self.vv_values[name])))
        for rid, v in self.coef_values.items():
            if v != self.coef_orig[rid]:
                r = self.rows[rid]
                edits.append((r.term.start, r.term.end, self._term_text(r, value=v)))
        return self._apply(edits)

    def compute_text(self) -> str:
        """Text for pycalphad: live coefficients become symbols (defined as FUNCTIONs at the end)."""
        # Values are pycalphad parameters, so the text holds placeholders (0): the text, and with it the compiled
        # functions, then depend only on which numbers are symbolic, not on their values.
        edits = [(d.expr_start, d.expr_end, "0") for d in self.vvdefs.values()]
        for rid in sorted(self.live):
            r = self.rows[rid]
            edits.append((r.term.start, r.term.end, self._term_text(r, symbol=r.symbol)))
        text = self._apply(edits)
        if self.live:
            defs = "".join(f"\nFUNCTION {self.rows[rid].symbol} 1 0; 10000 N !" for rid in sorted(self.live))
            text = text.rstrip() + "\n$ tdb-tweak: edited coefficients as symbols" + defs + "\n"
        return text

    def save(self, path: Optional[str] = None) -> str:
        path = path or self.path
        if not path:
            raise ValueError("no path to save to")
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(self.render())
        return path

    def mark_saved(self):
        """After saving: the current values become the originals (the file now holds them)."""
        live, locked = set(self.live), set(self.locked)
        new = TDBDocument(self.render(), self.path)
        if len(new.rows) != len(self.rows):
            raise RuntimeError("saved file has a different set of editable numbers")
        self.__dict__.update(new.__dict__)
        # rewriting a number keeps the term, so row ids are unchanged: keep symbolic coefficients symbolic so the
        # compiled models stay valid
        self.live = live
        self.locked = locked

    def phase_atoms(self, phase: str) -> float:
        """Atoms per formula unit: the site ratios of the sublattices that are not vacancies only."""
        if not hasattr(self, "_atoms"):
            ratios, cons = {}, {}
            for c in self.commands:
                if c.kind == "PHASE":
                    w = c.text.rstrip("!").split()
                    try:
                        n = int(w[3])
                        ratios[w[1].upper().split(":")[0]] = [float(x) for x in w[4:4 + n]]
                    except (IndexError, ValueError):
                        pass
                elif c.kind == "CONSTITUENT":
                    w = c.text.rstrip("!").split(None, 2)
                    if len(w) >= 3:
                        body = w[2].strip().strip(":")
                        cons[w[1].upper().split(":")[0]] = [[x.strip().strip("%#").upper() for x in sl.split(",")
                                                             if x.strip()] for sl in body.split(":")]
            self._atoms = {}
            for ph, rs in ratios.items():
                sls = cons.get(ph, [[]] * len(rs))
                self._atoms[ph] = sum(r for r, sl in zip(rs, sls) if [x for x in sl if x != "VA"] or not sl) or 1.0
        return self._atoms.get(phase.upper(), 1.0)

    def masses(self) -> Dict[str, float]:
        return dict(self.elements)
