"""Scan for common CALPHAD artifacts of a parameter set (things a fit can break far from where you look).

* a solid phase stable at very high temperature (above every melting point);
* the liquid splitting into two liquids at very high temperature;
* a solution phase that, on its own, has a miscibility gap at very high temperature ("inverted" gap, typically from
  large T-dependent interaction terms);
* the liquid stable at room temperature (a note: it can be real, e.g. Ga or Hg systems).
"""
from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from pycalphad import Workspace, variables as v

from .points import is_line_compound

HIGH_T = (4000.0, 6000.0)
GAP_T = (3000.0, 5000.0)
LOW_T = 300.0


@dataclass
class Finding:
    severity: str          # "warning" or "note"
    text: str


def _grid(els: Sequence[str], step: float):
    """Composition conditions over the whole system: {X(B): [...], X(C): [...]} as point lists."""
    if len(els) == 2:
        xs = np.round(np.arange(step / 2, 1.0, step), 6)
        return [{els[1]: float(x)} for x in xs]
    pts = []
    for b in np.arange(step / 2, 1.0, step):
        for c in np.arange(step / 2, 1.0 - b, step):
            pts.append({els[1]: float(round(b, 6)), els[2]: float(round(c, 6))})
    return pts


def _eq(engine, comps, phases, models, prf, params, T, comp):
    conds = {v.T: float(T), v.P: 101325.0, v.N: 1.0}
    for e, x in comp.items():
        conds[v.X(e)] = x
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wks = Workspace(engine.dbf, list(comps), list(phases), conds, models=models, phase_record_factory=prf,
                        parameters=params)
        eq = wks.eq
    ph = np.asarray(eq.Phase).reshape(-1)
    NP = np.asarray(eq.NP).reshape(-1)
    X = np.asarray(eq.X).reshape(len(ph), -1)
    return [(str(p), float(n), x) for p, n, x in zip(ph, NP, X) if str(p) and np.isfinite(n) and n > 1e-6]


def scan(engine, elements: Sequence[str], values: Dict[str, float], phases=None, step: float = None) -> List[Finding]:
    els = [e.upper() for e in elements]
    comps = tuple(sorted(set(els) | {"VA"}))
    phases = tuple(sorted(phases)) if phases else tuple(engine.phases_for(els))
    models, prf = engine._models(comps, phases, values)
    params = {k: float(values[k]) for k in engine.param_names}
    prf.update_parameters(params)
    step = step or (0.04 if len(els) == 2 else 0.1)
    grid = _grid(els, step)
    liquid = [p for p in phases if p.upper().startswith("LIQ") or p.upper() == "IONIC_LIQ"]
    out: List[Finding] = []

    def where(comp):
        return ", ".join(f"x({e.capitalize()})={x:.2f}" for e, x in comp.items())

    # 1, 2: what is stable at very high temperature
    for T in HIGH_T:
        solids, gaps = {}, []
        for comp in grid:
            try:
                res = _eq(engine, comps, phases, models, prf, params, T, comp)
            except Exception:
                continue
            for p, n, _ in res:
                base = p.split("#")[0]
                if base not in liquid:
                    solids.setdefault(base, comp)
            if sum(1 for p, n, _ in res if p.split("#")[0] in liquid) >= 2:
                gaps.append(comp)
        for p, comp in solids.items():
            out.append(Finding("warning", f"{p} is stable at {T:.0f} K ({where(comp)}): a solid above every "
                                          f"melting point"))
        if gaps:
            out.append(Finding("warning", f"the liquid splits into two liquids at {T:.0f} K ({where(gaps[0])}): "
                                          f"a high-temperature liquid miscibility gap"))
    # 3: each solution phase alone: inverted miscibility gaps
    for p in phases:
        if is_line_compound(models[p]):
            continue
        for T in GAP_T:
            hit = None
            for comp in grid:
                try:
                    res = _eq(engine, comps, [p], models, prf, params, T, comp)
                except Exception:
                    continue
                if len(res) >= 2:
                    Xs = np.array([x for _, _, x in res])
                    if np.max(np.ptp(Xs, axis=0)) > 0.05:
                        hit = comp
                        break
            if hit is not None:
                out.append(Finding("warning", f"{p} on its own has a miscibility gap at {T:.0f} K ({where(hit)}): "
                                              f"check its T-dependent interaction terms"))
                break
    # 4: liquid at room temperature
    for comp in grid:
        try:
            res = _eq(engine, comps, phases, models, prf, params, LOW_T, comp)
        except Exception:
            continue
        if any(p.split("#")[0] in liquid and n > 0.01 for p, n, _ in res):
            out.append(Finding("note", f"the liquid is stable at {LOW_T:.0f} K ({where(comp)})"))
            break
    return out


# ---------------------------------------------------------------------------------------------------------
# parameter magnitudes, per atom of the formula unit
# ---------------------------------------------------------------------------------------------------------
PER_ATOM_LIMIT = {"const": 400e3, "T": 100.0, "TLNT": 20.0, "T2": 0.05, "T3": 1e-5, "TINV": 1e7}
UNITS = {"const": "J/mol", "T": "J/mol/K", "TLNT": "J/mol/K", "T2": "J/mol/K²", "T3": "J/mol/K³", "TINV": "J·K/mol"}


def too_large(value: float, basis: str, atoms: float) -> bool:
    lim = PER_ATOM_LIMIT.get(basis)
    return lim is not None and abs(value) / max(atoms, 1e-9) > lim


def parameter_findings(doc, elements) -> List[Finding]:
    """Coefficients unusually large for the size of their phase's formula unit."""
    out, seen = [], set()
    for r in doc.rows_in(elements):
        if r.pname is None:
            continue
        val = doc.value(r) * (r.multiplier if r.kind == "VV" else 1.0)
        atoms = doc.phase_atoms(r.phase)
        if too_large(val, r.basis, atoms) and (r.symbol, r.basis) not in seen:
            seen.add((r.symbol, r.basis))
            lim = PER_ATOM_LIMIT[r.basis]
            out.append(Finding("warning", f"{(r.symbol + ' ') if r.kind == 'VV' else ''}{r.meaning()} = "
                                          f"{val:.4g} {UNITS[r.basis]}: {abs(val) / atoms:.3g} per atom "
                                          f"({atoms:g} atoms per formula unit) is beyond the usual "
                                          f"{lim:g} {UNITS[r.basis]}"))
    return out
