"""Move a binary invariant reaction (T and the three phase compositions) or a congruent point (T and the
common composition) to target values.

The invariant is solved directly (about 0.3 ms): its three composition sets come from a map, the phase with the
middle composition is fixed at zero amount, the overall composition is set between the other two, and T is
free.  pycalphad's solver then returns T and the three compositions for any parameter values (the composition
sets read the values through the shared PhaseRecordFactory, so nothing is recompiled).

Each iteration
--------------
1. residuals ``r`` = (T - T*)/tol_T and (x_i - x_i*)/tol_x for the targets in use (converged: all |r| <= 1);
2. forward-difference Jacobian ``J`` of the residuals with respect to the chosen parameters;
3. damped least squares in *limit-scaled* variables ``u_j = dp_j / limit_j`` (the most downhill change for the
   linearised problem, and the smallest one when there are more parameters than targets):
   ``u = -(Jsᵀ Js + λ I)⁻¹ Jsᵀ r``;
4. the step is shrunk so no parameter changes by more than its limit; if the invariant cannot be solved or the
   residual does not decrease, the step is halved and λ raised (Levenberg-Marquardt).

Per-iteration limit of a parameter: max(fraction x |value|, floor of its coefficient type), or a per-parameter
override.  The default floors correspond to about 50 kJ/mol at 1000 K.
"""
from __future__ import annotations

import copy
import math
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pycalphad import variables as v
from pycalphad.core.solver import Solver
from pycalphad.mapping import utils as map_utils
from pycalphad.mapping.primitives import Point
from pycalphad.mapping import zpf_equilibrium as zeq

from .engine import Cancelled, Engine, MapRequest
from .points import find_congruent_candidates, solve_congruent_from_line

DEFAULT_FLOORS = {"const": 50000.0, "T": 50.0, "TLNT": 7.0, "T2": 0.05, "T3": 5e-5, "TINV": 5e7}
OTHER_FLOOR = 1.0


@dataclass
class Variable:
    name: str                       # pycalphad parameter symbol (VV0003 or a live coefficient symbol)
    basis: str                      # const / T / TLNT / ...
    label: str = ""
    limit_override: Optional[float] = None   # absolute max change per iteration


@dataclass
class InvariantTarget:
    phases: Tuple[str, ...]                  # phase names as the map reports them (with #n multiplicity)
    T_guess: float
    T_target: Optional[float] = None
    x_target: Dict[str, Optional[float]] = field(default_factory=dict)   # phase -> x(B) target (mole fraction)
    T_tol: float = 1.0                       # K
    x_tol: float = 0.001                     # mole fraction
    kind: str = "invariant"                  # "invariant", "congruent" (binary) or "tietriangle" (ternary section)
    x_guess: float = 0.5                     # congruent: composition of the point to optimise
    corners_guess: Dict[str, Tuple[float, float]] = field(default_factory=dict)   # tietriangle: phase -> (xB, xC)


@dataclass
class OptSettings:
    max_iter: int = 10
    rel_limit: float = 1.0                   # fraction of |value|
    floors: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_FLOORS))
    fd_fraction: float = 1e-4                # finite-difference step as a fraction of the limit
    check_metastable: bool = True            # flag iterations where a target became metastable
    reject_metastable: bool = True           # ... and do not accept such steps (they are shortened instead)
    global_min_pdens: int = 500
    norm_weight: float = 0.0                 # optional penalty weight * sum((p - centre) / floor)^2; 0 = off
    norm_center: str = "zero"                # "zero" (small parameters) or "start" (stay near the start values)


@dataclass
class OptIteration:
    index: int
    values: Dict[str, float]
    points: List[Tuple[float, Dict[str, float]]]   # (T, x per phase) of every target
    residual: float                          # max |r| over all targets (<= 1: within tolerance)
    cost: float                              # sum r^2 of the targets
    clipped: bool = False                    # the step hit a parameter limit
    note: str = ""
    norm: float = 0.0                        # sum((p - centre) / floor)^2 of the varied parameters

    @property
    def T(self) -> float:                    # first target (single-target use)
        return self.points[0][0]

    @property
    def x(self) -> Dict[str, float]:
        return self.points[0][1]


@dataclass
class OptResult:
    iterations: List[OptIteration]
    converged: bool
    message: str
    seconds: float
    targets: List[InvariantTarget]
    variables: List[Variable]

    @property
    def target(self) -> InvariantTarget:
        return self.targets[0]


def target_label(t: InvariantTarget) -> str:
    return (" = " if t.kind == "congruent" else " + ").join(t.phases)


def limit_for(var: Variable, value: float, s: OptSettings) -> float:
    if var.limit_override is not None and var.limit_override > 0:
        return float(var.limit_override)
    floor = s.floors.get(var.basis, OTHER_FLOOR)
    return max(s.rel_limit * abs(value), floor)


class _Invariant:
    """Direct solver for one binary three-phase invariant."""

    def __init__(self, node, B_index: int, B: str):
        self.names = list(node.stable_phases_with_multiplicity)
        self.css0 = copy.deepcopy(node.stable_composition_sets)
        self.css = copy.deepcopy(self.css0)
        self.B_index = B_index
        self.B = B
        self.last_conditions = None
        self.last_mu = None

    def solve(self, start=None):
        css = copy.deepcopy(start if start is not None else self.css)
        xb = np.array([float(c.X[self.B_index]) for c in css])
        order = np.argsort(xb)
        for i, c in enumerate(css):
            c.fixed = bool(i == order[1])
            map_utils.update_cs_phase_frac(c, 0.0 if c.fixed else 0.5)
        conds = {v.P: np.array(101325.0), v.N: np.array(1.0),
                 v.X(self.B): np.array([0.5 * (xb[order[0]] + xb[order[2]])])}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = Solver(remove_metastable=False).solve(css, conds)
        except Exception:
            return None
        if not res.converged:
            return None
        sv = [str(s) for s in css[0].phase_record.state_variables]
        T = float(css[0].dof[sv.index("T")])
        x = {n: float(c.X[self.B_index]) for n, c in zip(self.names, css)}
        if not np.isfinite(T) or T <= 0 or not all(np.isfinite(list(x.values()))):
            return None
        if any(float(c.NP) < -1e-6 for c in css):
            return None
        conds[v.T] = np.array(T)
        return T, x, css, conds, np.array(res.chemical_potentials)

    def solve_robust(self):
        out = self.solve()
        if out is None:
            out = self.solve(self.css0)
        return out

    def accept(self, out):
        self.css = out[2]


class _CongruentPoint:
    """Adapter giving a congruent-point solver the interface of :class:`_Invariant`."""

    def __init__(self, solver, x0: float, T0: float, B: str):
        self.c = solver
        self.names = solver.names
        self.B = B
        self.guess = (x0, T0)
        self.guess0 = (x0, T0)
        self.css = None

    def _wrap(self, out):
        if out is None:
            return None
        T, x, css, mu = out
        conds = {v.P: np.array(101325.0), v.N: np.array(1.0), v.T: np.array(T), v.X(self.B): np.array([x])}
        return T, {n: x for n in self.names}, css, conds, mu

    def solve(self, start=None):
        return self._wrap(self.c.solve(self.guess[0], self.guess[1]))

    def solve_robust(self):
        out = self.solve()
        if out is None:
            out = self._wrap(self.c.solve(*self.guess0))
        return out

    def accept(self, out):
        self.guess = (out[1][self.names[0]], out[0])


def _residuals(T, x, tgt: InvariantTarget) -> np.ndarray:
    """x and tgt.x_target share keys: a phase name (binary) or "PHASE|EL" (a ternary corner's x(EL))."""
    r = []
    if tgt.T_target is not None:
        r.append((T - tgt.T_target) / tgt.T_tol)
    for k, xt in tgt.x_target.items():
        if xt is not None:
            r.append((x[k] - xt) / tgt.x_tol)
    return np.array(r, dtype=float)


class _TieTriangle:
    """Three-phase equilibrium of an isothermal section: at fixed T and P, with the overall composition inside the
    triangle, the three corners have no degree of freedom left."""

    def __init__(self, node, elements: Sequence[str], T: float):
        self.names = list(node.stable_phases_with_multiplicity)
        self.css0 = copy.deepcopy(node.stable_composition_sets)
        self.css = copy.deepcopy(self.css0)
        self.els = sorted(elements)
        self.B, self.C = elements[1], elements[2]
        self.iB, self.iC = self.els.index(self.B), self.els.index(self.C)
        self.T = float(T)

    def solve(self, start=None):
        css = copy.deepcopy(start if start is not None else self.css)
        cen = np.mean([[float(np.asarray(c.X)[self.iB]), float(np.asarray(c.X)[self.iC])] for c in css], axis=0)
        for c in css:
            c.fixed = False
            map_utils.update_cs_phase_frac(c, 1.0 / 3.0)
        conds = {v.P: np.array(101325.0), v.N: np.array(1.0), v.T: np.array(self.T),
                 v.X(self.B): np.array([cen[0]]), v.X(self.C): np.array([cen[1]])}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = Solver(remove_metastable=False).solve(css, conds)
        except Exception:
            return None
        if not res.converged or any(float(c.NP) < -1e-6 for c in css):
            return None
        x = {}
        for n, c in zip(self.names, css):
            X = np.asarray(c.X)
            x[f"{n}|{self.B}"] = float(X[self.iB])
            x[f"{n}|{self.C}"] = float(X[self.iC])
        if not all(np.isfinite(list(x.values()))):
            return None
        return self.T, x, css, conds, np.array(res.chemical_potentials)

    def solve_robust(self):
        out = self.solve()
        if out is None:
            out = self.solve(self.css0)
        return out

    def accept(self, out):
        self.css = out[2]


def states_from_compositions(engine: Engine, comps, models, prf, names: Sequence[str], guess: Dict[str, Sequence[float]],
                             indep: Sequence[str], T: float):
    """Starting composition sets for phases at given compositions (when the map does not have the feature).

    Each phase is sampled at T; the lowest-energy sample near the wanted composition is the start (this keeps
    e.g. an ordered and a disordered state of one phase apart)."""
    from types import SimpleNamespace
    from pycalphad import calculate
    from pycalphad.core.composition_set import CompositionSet
    els = sorted(c for c in comps if c != "VA")
    idx = [els.index(e) for e in indep]
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for n in names:
            base = n.split("#")[0]
            pr = prf.get(base)
            r = calculate(engine.dbf, list(comps), [base], T=T, P=101325.0, N=1, pdens=2000, model=models,
                          phase_records=prf, output="GM", to_xarray=False)
            X = np.asarray(r.X).reshape(-1, len(els))[:, idx]
            Y = np.asarray(r.Y).reshape(X.shape[0], -1)
            G = np.asarray(r.GM).ravel()
            d = np.linalg.norm(X - np.asarray(guess[n], dtype=float), axis=1)
            near = d <= d.min() + 0.02
            k = int(np.argmin(np.where(near, G, np.inf)))
            cs = CompositionSet(pr)
            conds = {v.P: np.array(101325.0), v.N: np.array(1.0), v.T: np.array(float(T))}
            cs.update(np.array(Y[k, :pr.phase_dof], dtype=float, copy=True), 1.0,
                      map_utils.get_statevars_array(conds, pr.state_variables))
            out.append(cs)
    return SimpleNamespace(stable_phases_with_multiplicity=list(names), stable_composition_sets=out)


def find_triangle_node(strategy, phases: Sequence[str], elements: Sequence[str], guess: Dict[str, Tuple[float, float]]):
    want = sorted(phases)
    els = sorted(elements)
    iB, iC = els.index(elements[1]), els.index(elements[2])
    best, best_d = None, math.inf
    for node in strategy.node_queue.nodes:
        if len(node.stable_composition_sets) != 3 or sorted(node.stable_phases_with_multiplicity) != want:
            continue
        d = 0.0
        for n, c in zip(node.stable_phases_with_multiplicity, node.stable_composition_sets):
            if n in guess:
                X = np.asarray(c.X)
                d += abs(X[iB] - guess[n][0]) + abs(X[iC] - guess[n][1])
        if d < best_d:
            best, best_d = node, d
    if best is not None and guess and best_d > 0.05:
        return None              # a triangle of the same phases elsewhere: build from the wanted corners instead
    return best


def find_invariant_node(strategy, phases: Sequence[str], T_guess: float):
    want = sorted(phases)
    best, best_d = None, math.inf
    for node in strategy.node_queue.nodes:
        if len(node.stable_composition_sets) != 3:
            continue
        if sorted(node.stable_phases_with_multiplicity) != want:
            continue
        d = abs(float(np.squeeze(node.global_conditions[v.T])) - T_guess)
        if d < best_d:
            best, best_d = node, d
    return best


def optimize_invariant(engine: Engine, req: MapRequest, values: Dict[str, float], variables: Sequence[Variable],
                       targets, settings: OptSettings, cancel: Optional[threading.Event] = None,
                       progress: Optional[Callable[[OptIteration], None]] = None) -> OptResult:
    """Move one or several binary invariants / congruent points to their targets at the same time.

    ``targets`` is an InvariantTarget or a list of them.  A target whose wanted values equal its current ones holds
    that point in place while the others move."""
    t0 = time.perf_counter()
    targets = [targets] if isinstance(targets, InvariantTarget) else list(targets)
    if req.kind not in ("binary", "ternary"):
        raise ValueError("optimisation works on binary diagrams and ternary isothermal sections")
    if req.kind == "ternary" and any(t.kind != "tietriangle" for t in targets):
        raise ValueError("on a ternary section, the targets are tie triangles")
    if req.kind == "binary" and any(t.kind == "tietriangle" for t in targets):
        raise ValueError("tie triangles are targets of ternary isothermal sections")
    if not variables:
        raise ValueError("choose at least one parameter to vary")
    if not targets:
        raise ValueError("add at least one target")
    for t in targets:
        if t.T_target is None and not any(xv is not None for xv in t.x_target.values()):
            raise ValueError(f"{target_label(t)}: tick at least one target value (temperature or a composition)")
    A, B = req.elements[0], req.elements[1]
    values = dict(values)
    names = [var.name for var in variables]
    missing = [n for n in names if n not in values]
    if missing:
        raise ValueError(f"not parameters of the compute database: {missing}")

    # locate every point on a map with the current values
    engine.run(req, values, cancel=cancel)
    strategy = engine.last_strategy
    comps = tuple(sorted(set(req.elements) | {"VA"}))
    phases = tuple(sorted(req.phases)) if req.phases else tuple(engine.phases_for(list(req.elements)))
    models, prf = engine._models(comps, phases, values)
    solvers = []
    for t in targets:
        if t.kind == "tietriangle":
            node = find_triangle_node(strategy, t.phases, req.elements, t.corners_guess)
            if node is None and t.corners_guess:
                node = states_from_compositions(engine, comps, models, prf, t.phases, t.corners_guess,
                                                req.elements[1:], req.temperature)
            if node is None:
                raise ValueError(f"the tie triangle {target_label(t)} is not on the current section")
            solvers.append(_TieTriangle(node, req.elements, req.temperature))
        elif t.kind == "congruent":
            cands = [c for c in find_congruent_candidates(strategy, B) if sorted(c[0]) == sorted(t.phases)]
            if not cands:
                raise ValueError(f"the congruent point {target_label(t)} is not on the current map")
            cand = min(cands, key=lambda c: abs(c[3] - t.T_guess) / 100 + abs(c[2] - t.x_guess))
            got = solve_congruent_from_line(*cand, A, B, models)
            if got is None:
                raise ValueError(f"could not solve the congruent point {target_label(t)} at the current values")
            solvers.append(_CongruentPoint(got[0], got[1][1], got[1][0], B))
        else:
            node = find_invariant_node(strategy, t.phases, t.T_guess)
            if node is not None and abs(float(np.squeeze(node.global_conditions[v.T])) - t.T_guess) > 20:
                node = None
            if node is None and t.corners_guess:
                node = states_from_compositions(engine, comps, models, prf, t.phases, t.corners_guess, [B], t.T_guess)
            if node is None:
                raise ValueError(f"the invariant {target_label(t)} is not on the current map")
            solvers.append(_Invariant(node, sorted([A, B]).index(B), B))
    system_info = strategy.system_info

    def set_values(vals):
        prf.update_parameters({k: float(vals[k]) for k in engine.param_names})

    def evaluate(vals, robust=False):
        set_values(vals)
        outs = []
        for s in solvers:
            o = s.solve_robust() if robust else s.solve()
            if o is None and not robust:
                o = s.solve_robust()
            if o is None:
                return None
            outs.append(o)
        return outs

    def residuals(outs):
        return np.concatenate([_residuals(o[0], o[1], t) for o, t in zip(outs, targets)])

    def accept(outs):
        for s, o in zip(solvers, outs):
            s.accept(o)

    def pts(outs):
        return [(float(o[0]), dict(o[1])) for o in outs]

    def metastable(outs):
        """[(target label, more stable phase)] for targets that are no longer the stable state."""
        found_all = []
        for t, o in zip(targets, outs):
            T, x, css, conds, mu = o
            if mu is None:
                continue
            try:
                pt = Point(conds, mu, [c for c in css if c.fixed], [c for c in css if not c.fixed])
                found = zeq._find_global_min_cs(pt, system_info, pdens=settings.global_min_pdens,
                                                tol=1e-4, num_candidates=200)
            except Exception:
                found = None
            if found is not None:
                found_all.append((target_label(t), found[0].phase_record.phase_name))
        return found_all

    floors = np.array([settings.floors.get(var.basis, OTHER_FLOOR) for var in variables])
    wn = math.sqrt(max(settings.norm_weight, 0.0))

    def norm_rows(pv):
        if wn == 0.0:
            return np.zeros(0)
        return wn * (pv - centre) / floors

    p = np.array([values[n] for n in names], dtype=float)
    centre = p.copy() if settings.norm_center == "start" else np.zeros_like(p)
    cur = evaluate(values, robust=True)
    if cur is None:
        set_values(values)
        raise ValueError("could not solve every target at the current values")
    accept(cur)
    r = residuals(cur)
    start_meta = {lab for lab, _ in metastable(cur)} if settings.check_metastable else set()
    start_note = "start" + ("; already metastable: " + ", ".join(sorted(start_meta)) if start_meta else "")
    hist = [OptIteration(0, dict(zip(names, p)), pts(cur), float(np.max(np.abs(r))), float(r @ r), note=start_note,
                         norm=float(np.sum(((p - centre) / floors) ** 2)))]
    if progress:
        progress(hist[-1])
    lam = 1e-3
    converged = bool(np.max(np.abs(r)) <= 1.0)
    message = "already within tolerance" if converged else ""
    try:
        for it in range(1, settings.max_iter + 1):
            if converged and not wn:
                break
            if cancel is not None and cancel.is_set():
                raise Cancelled()
            lim = np.array([limit_for(var, pv, settings) for var, pv in zip(variables, p)])
            base = dict(values)
            base.update(zip(names, p))
            # Jacobian in limit-scaled variables
            J = np.zeros((len(r), len(p)))
            for j in range(len(p)):
                h = settings.fd_fraction * lim[j]
                trial = dict(base)
                trial[names[j]] = p[j] + h
                out = evaluate(trial)
                if out is None:
                    trial[names[j]] = p[j] - h
                    out = evaluate(trial)
                    h = -h
                if out is None:
                    continue
                J[:, j] = (residuals(out) - r) / (h / lim[j])
            if not np.any(J):
                message = "the targets do not respond to the chosen parameters"
                break
            # optional norm rows: d(wn (p - c)/f)/du = wn * lim / f
            Rn = norm_rows(p)
            R = np.concatenate([r, Rn])
            JR = np.vstack([J, np.diag(wn * lim / floors)]) if wn else J
            accepted = False
            rejected_meta = []
            for attempt in range(10):
                JtJ = JR.T @ JR
                u = -np.linalg.solve(JtJ + lam * max(np.trace(JtJ) / len(p), 1e-12) * np.eye(len(p)), JR.T @ R)
                clipped = False
                m = float(np.max(np.abs(u)))
                if m > 1.0:
                    u = u / m
                    clipped = True
                p_new = p + u * lim
                trial = dict(values)
                trial.update(zip(names, p_new))
                out = evaluate(trial)
                if out is not None:
                    r_new = residuals(out)
                    Rn_new = norm_rows(p_new)
                    if r_new @ r_new + Rn_new @ Rn_new < R @ R:
                        meta = [m for m in metastable(out) if m[0] not in start_meta] \
                            if settings.reject_metastable else []
                        if not meta:
                            accepted = True
                            lam = max(lam / 3.0, 1e-9)
                            break
                        rejected_meta = meta
                lam *= 4.0
            if not accepted:
                if start_meta:
                    message = ("stopped: " + ", ".join(sorted(start_meta)) + " is already metastable at the current "
                               "values (another phase is more stable there), so it cannot be optimised as it is. "
                               "Map again, or pick the feature from a map made after choosing the parameters.")
                elif rejected_meta:
                    message = ("stopped: every step toward the targets makes "
                               + ", ".join(f"{lab} metastable ({ph} more stable)" for lab, ph in rejected_meta)
                               + " — add that feature as a target or vary other parameters")
                else:
                    message = "no further improvement (the targets cannot get closer with these parameters and limits)"
                break
            total_old = float(R @ R)
            p = p_new
            accept(out)
            r = r_new
            total_new = float(r @ r) + float(Rn_new @ Rn_new)
            note = "step limited" if clipped else ""
            if settings.check_metastable and not settings.reject_metastable:
                for lab, ph in metastable(out):
                    note = (note + "; " if note else "") + f"{lab} metastable: {ph} is more stable"
            hist.append(OptIteration(it, dict(zip(names, p)), pts(out), float(np.max(np.abs(r))), float(r @ r),
                                     clipped, note, norm=float(np.sum(((p - centre) / floors) ** 2))))
            if progress:
                progress(hist[-1])
            if np.max(np.abs(r)) <= 1.0:
                converged = True
                message = f"within tolerance after {it} iteration{'s' if it > 1 else ''}"
                if wn:
                    message += " (norm term on: continuing while it lowers the total)"
                    if total_old - total_new <= 1e-3 * total_old:
                        message = f"within tolerance, norm minimised after {it} iterations"
                        break
            elif wn:
                converged = False
        else:
            if not converged:
                message = f"stopped after {settings.max_iter} iterations (max |residual| {np.max(np.abs(r)):.3g} tolerances)"
            elif wn:
                message = f"within tolerance; the norm was still decreasing after {settings.max_iter} iterations"
    finally:
        set_values({**values, **dict(zip(names, p))})
    return OptResult(hist, converged, message, time.perf_counter() - t0, targets, list(variables))
