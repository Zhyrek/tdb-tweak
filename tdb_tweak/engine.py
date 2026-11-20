"""Phase-diagram mapping that recompiles only when the *structure* of the database changes.

Every editable number is a pycalphad parameter (a symbol whose value is passed at run time), so the models and
their compiled energy functions are built once per (database structure, components, phases).  A value edit
only writes into ``PhaseRecordFactory.param_values`` and the next map uses it.

pycalphad's mapping strategies take ``models`` and ``phase_record_factory`` but start their starting-point
equilibria with ``Workspace(..., models=..., phase_record_factory=...)`` and no ``parameters``; that Workspace
then calls ``update_parameters({})`` and fails with "Parameter symbol mismatch" (0.12), or uses the stored
values (0.11).  :func:`_install_patch` passes the factory's current values to it.
"""
from __future__ import annotations

import copy
import hashlib
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pycalphad import Database, variables as v
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.core.utils import filter_phases, get_state_variables, instantiate_models, unpack_species
import pycalphad.mapping.starting_points as _sp
import pycalphad.mapping.strategy.strategy_base as _sb
from pycalphad.mapping import IsoplethStrategy, TielineStrategy
from pycalphad.mapping.primitives import ZPFState
from pycalphad.mapping.strategy.tieline_strategy import _get_phase_specific_variable

try:
    import pycalphad_cg as _cg
except Exception:  # optional
    _cg = None

_patch_lock = threading.Lock()
_patched = False


def _install_patch():
    global _patched
    with _patch_lock:
        if _patched:
            return
        original = _sp.point_from_equilibrium

        def point_from_equilibrium(dbf, components, phases, conditions, **eq_kwargs):
            prf = eq_kwargs.get("phase_record_factory")
            if prf is not None and len(prf.param_symbols) and "parameters" not in eq_kwargs:
                eq_kwargs["parameters"] = {str(s): float(x) for s, x in zip(prf.param_symbols, prf.param_values)}
            return original(dbf, components, phases, conditions, **eq_kwargs)

        # every module that imported the function keeps its own reference
        import importlib
        import pkgutil
        import pycalphad.mapping as _mapping
        for info in pkgutil.walk_packages(_mapping.__path__, _mapping.__name__ + "."):
            try:
                mod = importlib.import_module(info.name)
            except Exception:
                continue
            if mod is not _sp and getattr(mod, "point_from_equilibrium", None) is original:
                mod.point_from_equilibrium = point_from_equilibrium
        _sb.point_from_equilibrium = point_from_equilibrium
        point_from_equilibrium.original = original
        _patched = True


def cg_available() -> bool:
    return _cg is not None


class Cancelled(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------
# requests and results
# ---------------------------------------------------------------------------------------------------------
@dataclass
class MapRequest:
    kind: str                                  # "binary", "ternary", "isopleth"
    elements: Tuple[str, ...]                  # binary (A, B): x = X(B); ternary (A, B, C): X(B), X(C);
                                               # isopleth (A, B, C): x = X(B), X(C) fixed
    phases: Optional[Tuple[str, ...]] = None   # None: all phases of the subsystem
    tmin: float = 300.0
    tmax: float = 2000.0
    tstep: float = 10.0
    temperature: float = 1000.0                # ternary isothermal section
    xmin: float = 0.0
    xmax: float = 1.0
    xstep: float = 0.01
    ymin: float = 0.0                          # ternary: X(C) range
    ymax: float = 1.0
    fixed: float = 0.1                         # isopleth: X(C)
    pressure: float = 101325.0
    use_cg: bool = False
    solver: str = ""                           # "auto": time both solvers now and then, use the faster
    global_min_pdens: int = 500
    global_check_interval: int = 1


@dataclass
class Boundary:
    phase: str
    x: np.ndarray
    y: np.ndarray
    partner: str = ""                          # the other phase of the two-phase region (tie-line strategies)
    partner_x: Optional[np.ndarray] = None     # its x at the same points (binary: composition; ternary: X(B))
    partner_y: Optional[np.ndarray] = None     # ternary: its X(C) at the same points


@dataclass
class MapResult:
    request: MapRequest
    boundaries: List[Boundary] = field(default_factory=list)          # phase boundary curves
    tielines: List[np.ndarray] = field(default_factory=list)          # each (n_points, 2 phases, 2 xy)
    invariants: List[Tuple[List[str], np.ndarray, np.ndarray]] = field(default_factory=list)
    congruent: List = field(default_factory=list)                       # points.CongruentPoint (binary)
    phase_space: Dict = field(default_factory=dict)                     # phase -> points.composition_space
    zpf: Optional[object] = None                                        # zpfscore.ZPFScore of the loaded data
    thermo: Optional[object] = None                                     # thermo.ThermoScore of thermochemical data
    warm: bool = False                          # started from the previous map's starting points (fast redraw)
    complete: bool = False
    seconds: float = 0.0
    compile_seconds: float = 0.0
    n_lines: int = 0
    phases: List[str] = field(default_factory=list)
    message: str = ""


# ---------------------------------------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------------------------------------
class Engine:
    """Holds the compute database and compiled models; not thread safe (use from one worker thread)."""

    def __init__(self):
        _install_patch()
        self._text_hash: Optional[str] = None
        self.dbf: Optional[Database] = None
        self._cache: Dict[tuple, Tuple[dict, PhaseRecordFactory]] = {}
        self.param_names: Tuple[str, ...] = ()
        self._starts: Dict[tuple, list] = {}       # request -> starting points of its last full map

    MAX_DATABASES = 4

    def set_database(self, compute_text: str, param_names: Sequence[str]) -> bool:
        """Make a compute database the active one; returns True when it was new (its models compile on first use).

        A few databases stay loaded side by side (e.g. the edited file and a reference file to compare with), each
        with its own compiled models, so switching between them costs nothing."""
        h = hashlib.sha256(compute_text.encode()).hexdigest()
        names = tuple(sorted(param_names))
        key = (h, names)
        if self._text_hash is not None and key == (self._text_hash, self.param_names):
            return False
        if not hasattr(self, "_dbs"):
            self._dbs = {}
        if self._text_hash is not None:
            self._dbs[(self._text_hash, self.param_names)] = (self.dbf, self._cache, self._starts)
        if key in self._dbs:
            self.dbf, self._cache, self._starts = self._dbs.pop(key)
            self._dbs[key] = (self.dbf, self._cache, self._starts)        # most recently used last
            self._text_hash, self.param_names = h, names
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.dbf = Database.from_string(compute_text, fmt="tdb")
        self._cache, self._starts = {}, {}
        self._text_hash, self.param_names = h, names
        while len(self._dbs) >= self.MAX_DATABASES:
            self._dbs.pop(next(iter(self._dbs)))
        return True

    def phases_for(self, elements: Sequence[str]) -> List[str]:
        comps = sorted(set(e.upper() for e in elements) | {"VA"})
        return sorted(filter_phases(self.dbf, unpack_species(self.dbf, comps), list(self.dbf.phases)))

    def _models(self, comps: Tuple[str, ...], phases: Tuple[str, ...], values: Dict[str, float]):
        key = (comps, phases)
        hit = self._cache.get(key)
        if hit is None:
            params = {k: float(values[k]) for k in self.param_names}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                models = instantiate_models(self.dbf, list(comps), list(phases), parameters=params)
                conds = {v.T: 1000.0, v.P: 101325.0, v.N: 1.0}
                prf = PhaseRecordFactory(self.dbf, list(comps), get_state_variables(models, conds), models,
                                         parameters=params)
            hit = (models, prf)
            self._cache[key] = hit
        return hit

    @staticmethod
    def _req_key(req: MapRequest) -> tuple:
        from dataclasses import asdict
        return tuple(sorted((k, tuple(v_) if isinstance(v_, list) else v_) for k, v_ in asdict(req).items()))

    RECHECK_EVERY = 30

    def run(self, req: MapRequest, values: Dict[str, float], cancel: Optional[threading.Event] = None,
            progress: Optional[Callable[[MapResult], None]] = None, progress_interval: float = 0.4,
            warm: bool = False) -> MapResult:
        """Map ``req``; with ``req.solver == "auto"`` the faster of pycalphad's solver and column generation is used
        (both are timed on the first map of a diagram and every RECHECK_EVERY maps after)."""
        if req.solver != "auto" or _cg is None:
            from dataclasses import replace
            return self._run(replace(req, solver="", use_cg=req.use_cg and _cg is not None), values, cancel,
                             progress, progress_interval, warm)
        from dataclasses import replace
        if not hasattr(self, "_solver_pick"):
            self._solver_pick = {}
        key = self._req_key(replace(req, solver="", use_cg=False, temperature=0.0))
        pick = self._solver_pick.get(key)
        if pick is None or (pick[1] >= self.RECHECK_EVERY and not warm):
            times, results = {}, {}
            for cg in (False, True):
                r = self._run(replace(req, solver="", use_cg=cg), values, cancel, progress if not cg else None,
                              progress_interval, warm)
                times[cg], results[cg] = r.seconds, r
            best = min(times, key=times.get)
            self._solver_pick[key] = [best, 0]
            res = results[best]
            res.message = (res.message + "; " if res.message else "") + \
                f"auto: {'column generation' if best else 'pycalphad'} " \
                f"({times[True]:.2f} s cg vs {times[False]:.2f} s pycalphad)"
            res.request = req
            return res
        pick[1] += 1
        res = self._run(replace(req, solver="", use_cg=pick[0]), values, cancel, progress, progress_interval, warm)
        res.message = (res.message + "; " if res.message else "") + ("auto: cg" if pick[0] else "auto: pycalphad")
        res.request = req
        return res

    def _run(self, req: MapRequest, values: Dict[str, float], cancel: Optional[threading.Event] = None,
             progress: Optional[Callable[[MapResult], None]] = None, progress_interval: float = 0.4,
             warm: bool = False) -> MapResult:
        """Map ``req``.  ``warm``: start from the previous full map's starting points instead of searching the
        axis edges again (about 35-55 % faster; a phase region not connected to any of them would be missed)."""
        if self.dbf is None:
            raise RuntimeError("no database loaded")
        t0 = time.perf_counter()
        els = [e.upper() for e in req.elements]
        comps = tuple(sorted(set(els) | {"VA"}))
        phases = tuple(sorted(req.phases)) if req.phases else tuple(self.phases_for(els))
        fresh = (comps, phases) not in self._cache
        models, prf = self._models(comps, phases, values)
        prf.update_parameters({k: float(values[k]) for k in self.param_names})
        t_compile = time.perf_counter() - t0 if fresh else 0.0

        conds, strategy_cls = self._conditions(req, els)
        kw = dict(models=models, phase_record_factory=prf, GLOBAL_MIN_PDENS=int(req.global_min_pdens),
                  GLOBAL_CHECK_INTERVAL=int(req.global_check_interval))
        result = MapResult(req, phases=list(phases))
        ctx = _cg.cg_mode() if (req.use_cg and _cg is not None) else _nullcontext()
        with warnings.catch_warnings(), ctx:
            warnings.simplefilter("ignore")
            strategy = strategy_cls(self.dbf, list(comps), list(phases), conds, **kw)
            rkey = self._req_key(req)
            used_warm = False
            if warm and self._starts.get(rkey):
                for c_, dirn in self._starts[rkey]:
                    try:
                        strategy.add_nodes_from_conditions(c_, dirn)
                    except Exception:
                        pass
                used_warm = len(strategy.node_queue.nodes) > 0
            if not used_warm:
                strategy.generate_automatic_starting_points()
                self._starts[rkey] = [({k: float(np.squeeze(val)) for k, val in n.global_conditions.items()},
                                       n.axis_direction) for n in strategy.node_queue.nodes]
            last = time.perf_counter()
            finished = False
            while not finished:
                if cancel is not None and cancel.is_set():
                    raise Cancelled()
                finished = strategy.iterate()
                if progress is not None and time.perf_counter() - last > progress_interval:
                    last = time.perf_counter()
                    partial = self._extract(strategy, req, els, phases, final=False)
                    partial.seconds = last - t0
                    partial.compile_seconds = t_compile
                    progress(partial)
            strategy._mapping_complete = True
            self.last_strategy = strategy
            result = self._extract(strategy, req, els, phases, final=True)
        result.complete = True
        result.warm = used_warm
        result.seconds = time.perf_counter() - t0
        result.compile_seconds = t_compile
        return result

    def point(self, req: MapRequest, values: Dict[str, float], T: float, x: Dict[str, float]) -> List[Tuple]:
        """Equilibrium at one point with the current parameters: [(phase, fraction, {element: x})]."""
        from pycalphad import Workspace
        els = [e.upper() for e in req.elements]
        comps = tuple(sorted(set(els) | {"VA"}))
        phases = tuple(sorted(req.phases)) if req.phases else tuple(self.phases_for(els))
        models, prf = self._models(comps, phases, values)
        params = {k: float(values[k]) for k in self.param_names}
        prf.update_parameters(params)
        conds = {v.T: float(T), v.P: req.pressure, v.N: 1.0}
        for e, val in x.items():
            conds[v.X(e)] = float(val)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wks = Workspace(self.dbf, list(comps), list(phases), conds, models=models, phase_record_factory=prf,
                            parameters=params)
            eq = wks.eq
        ph = np.asarray(eq.Phase).reshape(-1)
        NP = np.asarray(eq.NP).reshape(-1)
        X = np.asarray(eq.X).reshape(len(ph), -1)
        order = sorted(e for e in els)
        out = []
        for p_, n_, x_ in zip(ph, NP, X):
            if str(p_) and np.isfinite(n_) and n_ > 1e-8:
                out.append((str(p_), float(n_), {e: float(xx) for e, xx in zip(order, x_)}))
        return out

    def _system(self, req: MapRequest, values: Dict[str, float]):
        els = [e.upper() for e in req.elements]
        comps = tuple(sorted(set(els) | {"VA"}))
        phases = tuple(sorted(req.phases)) if req.phases else tuple(self.phases_for(els))
        models, prf = self._models(comps, phases, values)
        params = {k: float(values[k]) for k in self.param_names}
        prf.update_parameters(params)
        return els, comps, phases, models, prf, params

    def gx_curves(self, req: MapRequest, values: Dict[str, float], T: float, pdens: int = 3000) -> Dict:
        """Binary G-x curves at T: each phase's lowest molar Gibbs energy at each composition, and the common
        tangents of the equilibrium along x.  Energies are relative to the straight line between the pure
        elements' lowest energies (so the differences are visible)."""
        from pycalphad import Workspace, calculate
        els, comps, phases, models, prf, params = self._system(req, values)
        A, B = els[0], els[1]
        iB = sorted([A, B]).index(B)
        curves = {}
        bins = np.linspace(0.0, 1.0, 401)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for ph in phases:
                r = calculate(self.dbf, list(comps), [ph], T=float(T), P=req.pressure, N=1, pdens=pdens,
                              model=models, phase_records=prf, output="GM", to_xarray=False)
                X = np.asarray(r.X).reshape(-1, 2)[:, iB]
                G = np.asarray(r.GM).ravel()
                ok = np.isfinite(X) & np.isfinite(G)
                X, G = X[ok], G[ok]
                if len(X) == 0:
                    continue
                if np.ptp(X) < 1e-9:                       # a line compound: one point
                    curves[ph] = (np.array([X[0]]), np.array([G.min()]))
                    continue
                idx = np.digitize(X, bins)
                xs, gs = [], []
                for b in np.unique(idx):
                    sel = idx == b
                    k = np.argmin(G[sel])
                    xs.append(X[sel][k])
                    gs.append(G[sel][k])
                o = np.argsort(xs)
                curves[ph] = (np.asarray(xs)[o], np.asarray(gs)[o])
            # reference line between the pure elements
            gA = min((g[0] for x, g in curves.values() if x[0] < 2e-3), default=0.0)
            gB = min((g[-1] for x, g in curves.values() if x[-1] > 1 - 2e-3), default=0.0)
            # equilibrium along x: common tangents
            xs = np.linspace(0.005, 0.995, 100)
            wks = Workspace(self.dbf, list(comps), list(phases),
                            {v.T: float(T), v.P: req.pressure, v.N: 1.0, v.X(B): xs}, models=models,
                            phase_record_factory=prf, parameters=params)
            eq = wks.eq
            PH = np.asarray(eq.Phase).reshape(len(xs), -1)
            NP = np.asarray(eq.NP).reshape(len(xs), -1)
            XX = np.asarray(eq.X).reshape(len(xs), PH.shape[1], -1)[:, :, iB]
            MU = np.asarray(eq.MU).reshape(len(xs), -1)
        tangents, seen = [], set()
        iA = 1 - iB
        for k in range(len(xs)):
            live = [j for j in range(PH.shape[1]) if str(PH[k, j]) and np.isfinite(NP[k, j]) and NP[k, j] > 1e-8]
            if len(live) < 2:
                continue
            pts = sorted((float(XX[k, j]), str(PH[k, j])) for j in live)
            key = tuple((round(x, 3), p) for x, p in pts)
            if key in seen:
                continue
            seen.add(key)
            mu = MU[k]
            (x1, p1), (x2, p2) = pts[0], pts[-1]
            g = lambda x: (1 - x) * mu[iA] + x * mu[iB]
            tangents.append((x1, g(x1), x2, g(x2), (p1, p2)))
        return {"T": float(T), "curves": curves, "ref": (gA, gB), "tangents": tangents, "elements": (A, B)}

    def step(self, req: MapRequest, values: Dict[str, float], comp: Dict[str, float], tmin: float, tmax: float,
             n: int = 300) -> Dict:
        """Phase fractions vs temperature at a fixed composition."""
        from pycalphad import Workspace
        els, comps, phases, models, prf, params = self._system(req, values)
        Ts = np.linspace(float(tmin), float(tmax), int(n))
        conds = {v.T: Ts, v.P: req.pressure, v.N: 1.0}
        for e, x in comp.items():
            conds[v.X(e)] = float(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wks = Workspace(self.dbf, list(comps), list(phases), conds, models=models, phase_record_factory=prf,
                            parameters=params)
            eq = wks.eq
        PH = np.asarray(eq.Phase).reshape(len(Ts), -1)
        NP = np.asarray(eq.NP).reshape(len(Ts), -1)
        frac: Dict[str, np.ndarray] = {}
        for k in range(len(Ts)):
            for j in range(PH.shape[1]):
                p_ = str(PH[k, j])
                if p_ and np.isfinite(NP[k, j]) and NP[k, j] > 0:
                    frac.setdefault(p_, np.zeros(len(Ts)))[k] += NP[k, j]
        return {"T": Ts, "fractions": frac, "composition": dict(comp)}

    @staticmethod
    def _conditions(req: MapRequest, els: List[str]):
        P = req.pressure
        if req.kind == "binary":
            a, b = els
            return ({v.T: (req.tmin, req.tmax, req.tstep), v.X(b): (req.xmin, req.xmax, req.xstep),
                     v.P: P, v.N: 1}, TielineStrategy)
        if req.kind == "ternary":
            a, b, c = els
            return ({v.T: req.temperature, v.X(b): (req.xmin, req.xmax, req.xstep),
                     v.X(c): (req.ymin, req.ymax, req.xstep), v.P: P, v.N: 1}, TielineStrategy)
        if req.kind == "isopleth":
            a, b, c = els
            return ({v.T: (req.tmin, req.tmax, req.tstep), v.X(b): (req.xmin, req.xmax, req.xstep),
                     v.X(c): req.fixed, v.P: P, v.N: 1}, IsoplethStrategy)
        raise ValueError(req.kind)

    # -- results ------------------------------------------------------------------------------------------
    @staticmethod
    def _extract(strategy, req: MapRequest, els: List[str], phases, final: bool) -> MapResult:
        res = MapResult(req, phases=list(phases))
        lines = strategy.zpf_lines if final else [z for z in strategy.zpf_lines
                                                  if z.status != ZPFState.NOT_FINISHED]
        res.n_lines = len(strategy.zpf_lines)
        if req.kind in ("binary", "ternary"):
            if req.kind == "binary":
                xv, yv = v.X(els[1]), v.T
            else:
                xv, yv = v.X(els[1]), v.X(els[2])
            xs, ys = copy.copy(xv), copy.copy(yv)
            for var in (xs, ys):
                if hasattr(var, "phase_name") and var.phase_name is None:
                    var.phase_name = "*"
            for z in lines:
                if len(z.points) == 0:
                    continue
                pts = []
                phs = list(z.stable_phases_with_multiplicity)
                for p in phs:
                    x = np.asarray(z.get_var_list(_get_phase_specific_variable(p, xs)), dtype=float)
                    y = np.asarray(z.get_var_list(_get_phase_specific_variable(p, ys)), dtype=float)
                    res.boundaries.append(Boundary(p, x, y))
                    pts.append((x, y))
                if len(phs) == 2:
                    for k in (0, 1):
                        bnd = res.boundaries[-2 + k]
                        bnd.partner, bnd.partner_x, bnd.partner_y = phs[1 - k], pts[1 - k][0], pts[1 - k][1]
                if len(pts) == 2:
                    res.tielines.append(np.stack([np.stack([pts[0][0], pts[0][1]], -1),
                                                  np.stack([pts[1][0], pts[1][1]], -1)], axis=1))
            for node in strategy.node_queue.nodes:
                if len(node.stable_composition_sets) == 3:
                    names, X, Y = [], [], []
                    for p in node.stable_phases_with_multiplicity:
                        names.append(p)
                        X.append(float(np.squeeze(node.get_property(_get_phase_specific_variable(p, xs)))))
                        Y.append(float(np.squeeze(node.get_property(_get_phase_specific_variable(p, ys)))))
                    res.invariants.append((names, np.array(X), np.array(Y)))
            if final:
                try:
                    from .points import composition_space
                    res.phase_space = {ph: composition_space(m, els[1:]) for ph, m in strategy.models.items()}
                except Exception as ex:
                    res.message = f"composition spaces: {ex}"
            if final and req.kind == "binary":
                try:
                    from .points import detect_congruent
                    res.congruent = detect_congruent(strategy, els[0], els[1], strategy.models)
                except Exception as ex:
                    res.message = f"congruent points: {ex}"
        else:  # isopleth
            xv, yv = v.X(els[1]), v.T
            for z in lines:
                if len(z.points) == 0:
                    continue
                res.boundaries.append(Boundary(z.fixed_phases[0], np.asarray(z.get_var_list(xv), dtype=float),
                                               np.asarray(z.get_var_list(yv), dtype=float)))
            if final:
                try:
                    for d in strategy.get_invariant_data(xv, yv):
                        res.invariants.append((list(d.phases), np.asarray(d.x, dtype=float).ravel(),
                                               np.asarray(d.y, dtype=float).ravel()))
                except Exception as ex:  # invariants are a nicety; keep the lines
                    res.message = f"isopleth invariants: {ex}"
        return res


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
