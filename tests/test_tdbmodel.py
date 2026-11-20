import os
import warnings

import pytest

from tdb_tweak.tdbmodel import TDBDocument, fmt_number, split_terms

HERE = os.path.dirname(__file__)
ROOT = os.path.dirname(HERE)
CUMG = os.path.join(ROOT, "examples", "Cu-Mg", "Cu-Mg-generated.tdb")
ALCRNI = os.path.join(HERE, "alcrni.tdb")

SYNTH = """\
$ synthetic file for parser tests
ELEMENT VA VACUUM 0 0 0 !
ELEMENT A FCC_A1 10.0 0 0 !
ELEMENT B FCC_A1 20.0 0 0 !
ELEMENT C FCC_A1 30.0 0 0 !
FUNCTION VV0000 1 -1000; 10000 N !
FUNCTION VV0001 1 2.5; 10000 N !
FUNCTION VV0002 1 7; 10000 N !
FUNCTION GHSERA 298.15 -100+3.5*T; 6000 N !
PHASE LIQUID % 1 1 !
CONSTITUENT LIQUID :A,B,C: !
PARAMETER G(LIQUID,A;0) 298.15 -100+3.5*T-1.2*T*LN(T); 3000 Y +5E-3*T**2-200+1E5*T**(-1); 6000 N !
PARAMETER L(LIQUID,A,B;0) 1 VV0000+VV0001*T; 10000 N !
PARAMETER L(LIQUID,A,B;1) 1 VV0002 + 3*T; 10000 N !
PARAMETER L(LIQUID,A,B,C;0) 1 -VV0002*T+12*T**3; 10000 N !
PARAMETER TC(LIQUID,A,B;0) 1 VV0002+50; 10000 N !
PARAMETER BMAGN(LIQUID,A,B;0) 1 +GHSERA#+0.5; 10000 N !
PARAMETER G(LIQUID,C;0) 1 +GHSERA#; 10000 N !  $ comment with ! inside? no
"""


def doc():
    return TDBDocument(SYNTH)


def test_terms_and_bases():
    terms = split_terms("-100+3.5*T-1.2*T*LN(T)+5E-3*T**2+T**(-1)*7-2*T**3")
    assert [t.basis for t in terms] == ["const", "T", "TLNT", "T2", "TINV", "T3"]
    assert [t.sign * t.num for t in terms] == [-100, 3.5, -1.2, 5e-3, 7, -2]


def test_rows_meanings_and_shared_vv():
    d = doc()
    vv = [r for r in d.rows if r.kind == "VV"]
    by = {}
    for r in vv:
        by.setdefault(r.symbol, []).append(r.meaning())
    assert by["VV0000"] == ["LIQUID, L0(A,B), a"]
    assert by["VV0001"] == ["LIQUID, L0(A,B), b"]
    # one symbol used three times -> three rows
    assert sorted(by["VV0002"]) == sorted(["LIQUID, L1(A,B), a", "LIQUID, L0(A,B,C), b  (×-1)", "LIQUID, TC0(A,B), a"])
    # two VV terms in one parameter -> two rows
    assert len([r for r in vv if r.owner_name == "L(LIQUID,A,B;0)"]) == 2


def test_multi_range_coefficients():
    d = doc()
    g = [r for r in d.rows if r.kind == "COEF" and r.owner_name == "G(LIQUID,A;0)"]
    assert [(r.range_index, r.letter) for r in g] == [(0, "a"), (0, "b"), (0, "c"), (1, "d"), (1, "a"), (1, "f")]
    assert "[298.15–3000 K]" in g[0].meaning()


def test_sort_orders():
    d = doc()
    rows = sorted(d.rows_in(["A", "B", "C"]), key=lambda r: r.phase_sort_key())
    labels = [r.pname.term_label() for r in rows]
    # G before every L, L0 before L1, TC after the L terms
    assert labels.index("G") < labels.index("L0") < labels.index("L1") < labels.index("TC0")
    l0 = [r.meaning() for r in rows if r.pname.term_label() == "L0"]
    assert l0[:2] == ["LIQUID, L0(A,B), a", "LIQUID, L0(A,B), b"]
    vv_sorted = [r.symbol for r in sorted(d.rows, key=lambda r: r.vv_sort_key()) if r.kind == "VV"]
    assert vv_sorted == sorted(vv_sorted, key=lambda s: int(s[2:]))
    # coefficient order a, b, c, d, e, f inside a parameter
    l0abc = [r.letter for r in rows if r.owner_name == "L(LIQUID,A,B,C;0)"]
    assert l0abc == ["b", "e"]


def test_subsystem_and_lower_order():
    d = doc()
    ab = d.rows_in(["A", "B"])
    assert all("C" not in r.elements for r in ab)
    abc = ["A", "B", "C"]
    lower = {r.meaning() for r in d.rows_in(abc) if d.is_lower_order(r, abc)}
    assert "LIQUID, L0(A,B), a" in lower
    assert not any("(A,B,C)" in m for m in lower)


def test_render_unchanged_is_identical():
    for p in (CUMG, ALCRNI):
        d = TDBDocument.from_file(p)
        assert d.render() == d.text


def test_edit_vv_and_coefficients():
    d = doc()
    r = next(r for r in d.rows if r.symbol == "VV0000")
    assert d.set_value(r, -2500.25) is False
    first = next(r for r in d.rows if r.kind == "COEF" and r.letter == "a" and r.range_index == 0)
    mid = next(r for r in d.rows if r.kind == "COEF" and r.letter == "b" and r.range_index == 0)
    assert d.set_value(first, 42) is True          # first term, written with its own sign
    d.set_value(mid, -0.75)                        # middle term, now negative
    out = d.render()
    assert "FUNCTION VV0000 1 -2500.25; 10000 N !" in out
    assert "PARAMETER G(LIQUID,A;0) 298.15 +42-0.75*T-1.2*T*LN(T);" in out
    # everything else untouched
    assert out.replace("-2500.25", "-1000").replace("+42-0.75*T", "-100+3.5*T") == SYNTH
    re_read = TDBDocument(out)
    assert re_read.vv_values["VV0000"] == -2500.25


def test_compute_text_does_not_depend_on_values():
    d = doc()
    c = next(r for r in d.rows if r.kind == "COEF")
    d.set_value(c, 1.0)
    t1 = d.compute_text()
    d.set_value(c, 2.0)
    d.set_value(next(r for r in d.rows if r.symbol == "VV0001"), 99.0)
    assert d.compute_text() == t1
    assert f"+{c.symbol}#" in t1 or f"{c.symbol}#" in t1


def test_pycalphad_reads_edits():
    from pycalphad import Database, variables as v
    d = TDBDocument.from_file(CUMG)
    r = next(r for r in d.rows if r.symbol == "VV0003")
    d.set_value(r, 1234.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db = Database.from_string(d.render(), fmt="tdb")
    assert float(db.symbols["VV0003"].subs({v.T: 1000})) == 1234.5


def test_mark_saved_keeps_live_coefficients(tmp_path):
    p = tmp_path / "a.tdb"
    p.write_text(SYNTH)
    d = TDBDocument.from_file(str(p))
    c = next(r for r in d.rows if r.kind == "COEF")
    d.set_value(c, 7.0)
    text_before = d.compute_text()
    d.save()
    d.mark_saved()
    assert d.modified_count() == 0
    assert c.rid in d.live
    assert d.compute_text() == text_before
    assert d.value(d.rows[c.rid]) == 7.0


@pytest.mark.parametrize("x", [0.0, 1.0, -32539.5, 1e-5, -2.74381, 123456789.0, 3.3e20, -1.5e-12])
def test_fmt_number_roundtrip(x):
    assert float(fmt_number(x)) == x
