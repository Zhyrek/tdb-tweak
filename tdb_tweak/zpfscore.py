"""ZPF score of ESPEI phase-equilibrium data against the current parameters (ESPEI's driving-force idea).

For every data tie-line at temperature T:

1. each vertex with a known composition: its phase alone, at internal equilibrium at that composition (a
   single-phase solve) gives chemical potentials μ_v;
2. the target hyperplane is the mean of those μ_v (ESPEI's ``estimate_hyperplane``); for a single-phase data
   point (one vertex) it is the chemical potentials of the full equilibrium at that composition and T instead, so
   the point scores 0 only if its phase really is the stable one there;
3. the driving force of each vertex is the distance of its phase from the hyperplane: for a known composition
   ``μ·x − G(x)``; for an unknown one (``null`` in the dataset) the largest ``μ·x − G`` over the phase's states.

A perfect fit has all driving forces 0.  The score is the RMS over all vertices, in J/mol.
"""
from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from pycalphad import variables as v
from pycalphad.core.solver import Solver
from pycalphad.mapping import utils as map_utils

from .points import _gm, is_line_compound
from .zpf import ZPFDataset, fractions_for, matching


@dataclass
class ZPFScore:
    rms: float                                    # J/mol over all vertices
    n: int
    per_dataset: Dict[str, tuple] = field(default_factory=dict)    # path -> (rms, n)
    worst: List[tuple] = field(default_factory=list)                # (|DF|, label) largest few
    skipped: int = 0

    def text(self) -> str:
        return f"ZPF {self.rms:.0f} J/mol ({self.n} pts)" if self.n else "ZPF: no matching data"


class _Phases:
    """Starting states and single-phase solves of the phases of one system (warm starts cached per phase)."""

    def __init__(self, engine, comps, models, prf, elements):
        self.engine, self.comps, self.models, self.prf = engine, comps, models, prf
        self.els = sorted(e for e in comps if e != "VA")
        self.indep = list(elements[1:])
        self.idx = [self.els.index(e) for e in self.indep]
        self._samples: Dict[tuple, tuple] = {}
        self.line = {p: is_line_compound(m) for p, m in models.items()}

    def samples(self, phase: str, T: float, pdens: int = 400):
        key = (phase, round(T, 1), pdens)
        if key not in self._samples:
            from pycalphad import calculate
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = calculate(self.engine.dbf, list(self.comps), [phase], T=T, P=101325.0, N=1, pdens=pdens,
                              model=self.models, phase_records=self.prf, output="GM", to_xarray=False)
            X = np.asarray(r.X).reshape(-1, len(self.els))
            Y = np.asarray(r.Y).reshape(X.shape[0], -1)
            G = np.asarray(r.GM).ravel()
            self._samples[key] = (X, Y, G)
        return self._samples[key]

    def equilibrium_mu(self, x: Dict[str, float], T: float):
        """Chemical potentials of the full equilibrium (all phases) at composition x and T."""
        from pycalphad import Workspace
        conds = {v.T: float(T), v.P: 101325.0, v.N: 1.0}
        for e in self.indep:
            conds[v.X(e)] = float(x[e])
        params = {k: float(self.prf.param_values[i]) for i, k in enumerate(self.prf.param_symbols)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wks = Workspace(self.engine.dbf, list(self.comps), list(self.models), conds, models=self.models,
                            phase_record_factory=self.prf, parameters={str(k): v_ for k, v_ in params.items()})
            mu = np.asarray(wks.eq.MU, dtype=float).ravel()
        return mu if np.all(np.isfinite(mu)) else None

    def single(self, phase: str, x: Dict[str, float], T: float):
        """(GM, mu over sorted elements or None, full composition array) of a phase alone at composition x."""
        from pycalphad.core.composition_set import CompositionSet
        pr = self.prf.get(phase)
        X, Y, G = self.samples(phase, 1000.0, pdens=400)          # starting site fractions (T-independent start)
        want = np.array([x[e] for e in self.indep])
        d = np.linalg.norm(X[:, self.idx] - want, axis=1)
        k = int(np.argmin(np.where(d <= d.min() + 0.02, G, np.inf)))
        cs = CompositionSet(pr)
        conds = {v.P: np.array(101325.0), v.N: np.array(1.0), v.T: np.array(float(T))}
        cs.update(np.array(Y[k, :pr.phase_dof], dtype=float, copy=True), 1.0,
                  map_utils.get_statevars_array(conds, pr.state_variables))
        if self.line.get(phase, False):
            return _gm(cs), None, np.asarray(cs.X, dtype=float)
        for e in self.indep:
            conds[v.X(e)] = np.array([float(x[e])])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = Solver(remove_metastable=False).solve([cs], conds)
        except Exception:
            return None
        if not res.converged:
            return None
        return _gm(cs), np.asarray(res.chemical_potentials, dtype=float), np.asarray(cs.X, dtype=float)


def driving_forces(engine, request, values: Dict[str, float], datasets: Sequence[ZPFDataset]):
    """Driving force (J/mol) of every data vertex in a fixed order: [(dataset path, label, df or nan)].

    The order and length depend only on the data, so the list can be used as optimiser residuals; a vertex that
    cannot be evaluated at some parameter values is nan there."""
    els = list(request.elements[:2] if request.kind == "binary" else request.elements[:3])
    data = [d for d in matching(datasets, els) if d.enabled]
    if not data:
        return []
    comps = tuple(sorted(set(els) | {"VA"}))
    phases = tuple(sorted(request.phases)) if request.phases else tuple(engine.phases_for(els))
    models, prf = engine._models(comps, phases, values)
    prf.update_parameters({k: float(values[k]) for k in engine.param_names})
    P = _Phases(engine, comps, models, prf, els)
    out = []
    for ds in data:
        for tl in ds.tielines:
            slots = []                                      # (vertex, base, fractions or None)
            for vx in tl.vertices:
                base = vx.phase.split("#")[0]
                if base not in models:
                    continue
                slots.append((vx, base, fractions_for(vx, els)))
            solved = {}
            for i, (vx, base, fr) in enumerate(slots):
                if fr is not None:
                    o = P.single(base, fr, tl.T)
                    if o is not None:
                        solved[i] = o
            if len(tl.vertices) == 1 and len(solved) == 1:
                eq_mu = P.equilibrium_mu(slots[next(iter(solved))][2], tl.T)
                mus = [eq_mu] if eq_mu is not None else []
            else:
                mus = [o[1] for o in solved.values() if o[1] is not None]
            mu = np.mean(mus, axis=0) if mus else None
            for i, (vx, base, fr) in enumerate(slots):
                where = f"{ds.label}: {vx.phase}{'' if fr is not None else ' (composition unknown)'} at {tl.T:g} K"
                if mu is None:
                    out.append((ds.path, where, float("nan")))
                elif fr is not None:
                    if i in solved:
                        G, _, Xfull = solved[i]
                        out.append((ds.path, where, float(mu @ Xfull - G)))
                    else:
                        out.append((ds.path, where, float("nan")))
                else:
                    X, Y, Gs = P.samples(base, tl.T, pdens=200)
                    out.append((ds.path, where, float(np.max(X @ mu - Gs))))
    return out


def zpf_score(engine, request, values: Dict[str, float], datasets: Sequence[ZPFDataset]) -> Optional[ZPFScore]:
    dfs = driving_forces(engine, request, values, datasets)
    if not dfs and not [d for d in matching(datasets, list(request.elements[:2] if request.kind == "binary"
                                                             else request.elements[:3])) if d.enabled]:
        return None
    good = [(p, w, d) for p, w, d in dfs if np.isfinite(d)]
    per: Dict[str, list] = {}
    for p, w, d in good:
        per.setdefault(p, []).append(d)
    per_ds = {p: (float(np.sqrt(np.mean(np.square(v)))), len(v)) for p, v in per.items()}
    if not good:
        return ZPFScore(0.0, 0, per_ds, [], len(dfs))
    worst = sorted(((abs(d), w) for p, w, d in good), key=lambda t: -t[0])[:8]
    return ZPFScore(float(np.sqrt(np.mean(np.square([d for _, _, d in good])))), len(good), per_ds, worst,
                    len(dfs) - len(good))
