"""Direct solvers for special points of binary diagrams (no mapping): congruent points.

A congruent point of phases α and β (congruent melting, or a congruent solid-state transformation) is where
both phases have the same composition at an extremum of their two-phase region.  There the chemical potentials
of both elements are equal in the two phases at the same composition x:

    G_α(x, T) = G_β(x, T)      and      dG_α/dx = dG_β/dx        (two equations, unknowns x and T)

with each phase at its own internal equilibrium at composition x (a single-phase solve at fixed x; dG/dx is
μ_B - μ_A of that solve).  If either phase is a line compound, x is its composition and only the first equation
remains (unknown T).
"""
from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize as sopt

from pycalphad import variables as v
from pycalphad.core.solver import Solver
from pycalphad.mapping import utils as map_utils


@dataclass
class CongruentPoint:
    phases: Tuple[str, str]
    x: float                    # mole fraction of B
    T: float
    fixed_x: bool               # a line compound fixes the composition
    compound: str = ""          # which phase is the line compound


def is_line_compound(model) -> bool:
    return all(len([c for c in sl if str(c) != "VA"]) <= 1 for sl in model.constituents)


def _gm(cs) -> float:
    out = np.zeros(1)
    cs.phase_record.prop_2d(out, np.ascontiguousarray(np.asarray(cs.dof, dtype=float)[None, :]), b"GM")
    return float(out[0])


class Congruent:
    def __init__(self, cs_a, cs_b, names: Sequence[str], A: str, B: str, line: Sequence[bool]):
        self.names = list(names)
        self.cs = [copy.deepcopy(cs_a), copy.deepcopy(cs_b)]
        for c in self.cs:
            c.fixed = False
            map_utils.update_cs_phase_frac(c, 1.0)
        self.cs0 = copy.deepcopy(self.cs)
        self.A, self.B = A, B
        self.iB = sorted([A, B]).index(B)
        self.line = list(line)
        self.xc = None
        for c, is_line in zip(self.cs, self.line):
            if is_line:
                self.xc = float(np.asarray(c.X)[self.iB])
        self.last_mu = None

    def _single(self, k: int, x: float, T: float):
        """Phase k alone at composition x and temperature T -> (GM, dG/dx, mu, cs) or None."""
        cs = copy.deepcopy(self.cs[k])
        sv = cs.phase_record.state_variables
        conds = {v.P: np.array(101325.0), v.N: np.array(1.0), v.T: np.array(float(T))}
        cs.update(np.asarray(cs.dof)[len(sv):], 1.0, map_utils.get_statevars_array(conds, sv))
        if self.line[k]:
            return _gm(cs), None, None, cs
        conds[v.X(self.B)] = np.array([float(x)])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = Solver(remove_metastable=False).solve([cs], conds)
        except Exception:
            return None
        if not res.converged:
            return None
        mu = np.asarray(res.chemical_potentials, dtype=float)
        iA = 1 - self.iB
        return _gm(cs), float(mu[self.iB] - mu[iA]), mu, cs

    def solve(self, x0: float, T0: float):
        """Returns (T, x, cs list, mu) or None."""
        if self.xc is not None:
            x = self.xc
            k_sol = [i for i in range(2) if not self.line[i]]

            def f(T):
                a = self._single(0, x, T)
                b = self._single(1, x, T)
                if a is None or b is None:
                    raise ValueError
                return a[0] - b[0]
            try:
                T = sopt.newton(f, T0, tol=1e-7, maxiter=60)
            except (RuntimeError, ValueError, OverflowError):
                return None
            parts = [self._single(i, x, T) for i in range(2)]
            if any(p is None for p in parts) or not np.isfinite(T) or T <= 0:
                return None
            mu = parts[k_sol[0]][2] if k_sol else None
            self.cs = [p[3] for p in parts]
            return float(T), float(x), self.cs, mu

        def F(z):
            x, T = z[0], z[1] * 1000.0
            if not (0.0 < x < 1.0) or T <= 0:
                return [1e6, 1e6]
            a = self._single(0, x, T)
            b = self._single(1, x, T)
            if a is None or b is None:
                return [1e6, 1e6]
            return [(a[0] - b[0]) / 1000.0, (a[1] - b[1]) / 1000.0]

        sol = sopt.root(F, [x0, T0 / 1000.0], method="hybr", options={"xtol": 1e-10})
        x, T = float(sol.x[0]), float(sol.x[1] * 1000.0)
        if not sol.success or not (0 < x < 1) or T <= 0 or max(abs(np.asarray(sol.fun))) > 1e-5:
            return None
        parts = [self._single(i, x, T) for i in range(2)]
        if any(p is None for p in parts):
            return None
        # both phases must keep their own identity: equal compositions *and* different states
        self.cs = [p[3] for p in parts]
        return T, x, self.cs, parts[0][2]


def find_congruent_candidates(strategy, B: str, dx_max: float = 0.02, edge: float = 0.005):
    """Two-phase ZPF lines whose phase compositions touch inside the composition range.

    Returns [(names, k, x, T, zpf_line)] - the point of each line with the smallest composition difference."""
    out = []
    for z in strategy.zpf_lines:
        names = list(z.stable_phases_with_multiplicity)
        if len(names) != 2 or len(z.points) < 2:
            continue
        xa = np.asarray(z.get_var_list(v.X(names[0], B)), dtype=float)
        xb = np.asarray(z.get_var_list(v.X(names[1], B)), dtype=float)
        T = np.asarray(z.get_var_list(v.T), dtype=float)
        dx = np.abs(xa - xb)
        k = int(np.nanargmin(dx))
        xm = 0.5 * (xa[k] + xb[k])
        if dx[k] <= dx_max and edge < xm < 1 - edge:
            out.append((names, k, float(xm), float(T[k]), z))
    return out


def solve_congruent_from_line(names, k, x, T, z, A: str, B: str, models) -> Optional[Tuple[Congruent, tuple]]:
    css = z.points[k].stable_composition_sets
    by = dict(zip(z.points[k].stable_phases_with_multiplicity, css))
    cs_a, cs_b = by.get(names[0]), by.get(names[1])
    if cs_a is None or cs_b is None:
        cs_a, cs_b = css[0], css[1]
    line = [is_line_compound(models[n.split("#")[0]]) for n in names]
    if all(line):
        return None              # two line compounds of one composition: not handled
    c = Congruent(cs_a, cs_b, names, A, B, line)
    out = c.solve(x, T)
    if out is None:
        return None
    return c, out


def detect_congruent(strategy, A: str, B: str, models) -> List[CongruentPoint]:
    pts = []
    for names, k, x, T, z in find_congruent_candidates(strategy, B):
        got = solve_congruent_from_line(names, k, x, T, z, A, B, models)
        if got is None:
            continue
        c, (Ts, xs, _, _) = got
        if abs(Ts - T) > 50 or abs(xs - x) > 0.05:
            continue             # converged to something else
        comp = next((n for n, l in zip(names, c.line) if l), "")
        pts.append(CongruentPoint((names[0], names[1]), xs, Ts, c.xc is not None, comp))
    return pts


# ---------------------------------------------------------------------------------------------------------
# reachable compositions of a phase (stoichiometry)
# ---------------------------------------------------------------------------------------------------------
def composition_space(model, indep: Sequence[str], n_random: int = 300) -> Dict:
    """Compositions a phase's sublattice model can reach, in the independent mole fractions ``indep``.

    Returns {"rank": 0|1|2, "origin": [...], "dir": [...] or None, "lo": float, "hi": float}:
    rank 0 - a fixed composition (line compound); rank 1 - a line, e.g. (Fe)(Ni,Cr) where x(Fe) is fixed and
    Ni/Cr exchange, with "dir" the unit direction and [lo, hi] the reachable range of the coordinate along it;
    rank 2 - an area (only for ternaries)."""
    import itertools

    subl = [list(sl) for sl in model.constituents]
    ratios = [float(r) for r in model.site_ratios]
    indep = [e.upper() for e in indep]

    def comp(ys):
        num, den = {}, 0.0
        for a, sl, y in zip(ratios, subl, ys):
            for sp, yy in zip(sl, y):
                for el, n in getattr(sp, "constituents", {str(sp): 1}).items():
                    el = str(el).upper()
                    if el == "VA":
                        continue
                    w = a * yy * float(n)
                    num[el] = num.get(el, 0.0) + w
                    den += w
        if den <= 0:
            return None
        return [num.get(e, 0.0) / den for e in indep]

    pts = []
    combos = itertools.islice(itertools.product(*[range(len(sl)) for sl in subl]), 2000)
    for idx in combos:                       # end members (vertices of the reachable set)
        ys = [np.eye(len(sl))[i] for sl, i in zip(subl, idx)]
        c = comp(ys)
        if c is not None:
            pts.append(c)
    n_vert = len(pts)
    rng = np.random.default_rng(0)
    for _ in range(n_random):
        c = comp([rng.dirichlet(np.ones(len(sl))) for sl in subl])
        if c is not None:
            pts.append(c)
    X = np.asarray(pts, dtype=float)
    o = X.mean(axis=0)
    if len(indep) == 0 or len(X) == 0:
        return {"rank": 0, "origin": o.tolist(), "dir": None, "lo": 0.0, "hi": 0.0}
    _, S, Vt = np.linalg.svd(X - o, full_matrices=False)
    rank = int(np.sum(S > 1e-7 * max(1.0, float(np.sqrt(len(X))))))
    if rank == 0:
        return {"rank": 0, "origin": o.tolist(), "dir": None, "lo": 0.0, "hi": 0.0}
    if rank == 1 and len(indep) >= 1:
        d = Vt[0] / np.linalg.norm(Vt[0])
        t = (X[:max(n_vert, 1)] - o) @ d
        return {"rank": 1, "origin": o.tolist(), "dir": d.tolist(), "lo": float(t.min()), "hi": float(t.max())}
    return {"rank": min(rank, len(indep)), "origin": o.tolist(), "dir": None, "lo": 0.0, "hi": 0.0}


def constrain(space: Optional[Dict], x: Sequence[float]) -> Optional[List[float]]:
    """Project a wanted composition onto what the phase can reach (None: it cannot move)."""
    if space is None:
        return list(x)
    r = space["rank"]
    if r == 0:
        return None
    if r == 1:
        o, d = np.asarray(space["origin"]), np.asarray(space["dir"])
        t = float(np.clip((np.asarray(x, float) - o) @ d, space["lo"], space["hi"]))
        return (o + t * d).tolist()
    return list(x)
