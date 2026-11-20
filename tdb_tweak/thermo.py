"""ESPEI thermochemical datasets (non-ZPF) and their score against the current parameters.

Supported, following ESPEI's definitions:

* fixed-configuration data (datasets with a ``solver`` key): HM/SM/CPM ``_MIX`` (relative to the phase's
  endmembers) and ``_FORM`` (relative to the elements in their SER reference phases, via pycalphad's
  ``shift_reference_state``), at the given sublattice configurations and occupancies;
* activities ``ACR_<EL>`` with a reference state: compared as chemical potentials, R·T·ln(a).

Model contributions are excluded as ESPEI does: a dataset's ``excluded_model_contributions``, or the ones its
``tags`` map to in the ESPEI input YAML (``system.tags``), found next to or above the datasets folder.  A tagged
dataset whose tags cannot be resolved is skipped (and counted).

Residuals are normalised by ESPEI's default standard deviations (HM 500 J/mol, SM and CPM 0.2 J/mol/K, chemical
potential 500 J/mol) divided by the dataset weight; the score is their RMS ("σ").
"""
from __future__ import annotations

import glob
import json
import os
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pycalphad import variables as v

SIGMA = {"HM": 500.0, "SM": 0.2, "CPM": 0.2, "ACR": 500.0}
R = 8.314462618


@dataclass
class ThermoDataset:
    path: str
    output: str
    components: List[str]
    phases: List[str]
    conditions: Dict
    values: np.ndarray
    reference: str = ""
    comment: str = ""
    solver: Optional[Dict] = None
    reference_state: Optional[Dict] = None
    weight: float = 1.0
    excluded: Optional[Tuple[str, ...]] = None       # None: unresolved tags (skipped)
    disabled: bool = False
    enabled: bool = True

    @property
    def elements(self) -> frozenset:
        return frozenset(c.upper() for c in self.components if c.upper() != "VA")

    @property
    def label(self) -> str:
        return self.reference or os.path.basename(self.path)

    @property
    def kind(self) -> str:
        return "fixed" if self.solver is not None else ("activity" if self.output.startswith("ACR") else "other")


@dataclass
class ThermoScore:
    rms: float                      # RMS of normalised residuals (σ)
    n: int
    per_dataset: Dict[str, tuple] = field(default_factory=dict)
    worst: List[tuple] = field(default_factory=list)
    skipped: int = 0

    def text(self) -> str:
        return f"thermo {self.rms:.2g}σ ({self.n} pts)" if self.n else "thermo: no usable data"


def find_tag_map(folder: str) -> Dict[str, Dict]:
    """``system.tags`` of an ESPEI input YAML in the folder or up to two levels above."""
    try:
        import yaml
    except ImportError:
        return {}
    d = os.path.abspath(folder)
    for _ in range(3):
        for f in sorted(glob.glob(os.path.join(d, "*.yaml")) + glob.glob(os.path.join(d, "*.yml"))):
            try:
                cfg = yaml.safe_load(open(f)) or {}
            except Exception:
                continue
            tags = (cfg.get("system") or {}).get("tags")
            if isinstance(tags, dict):
                return tags
        d = os.path.dirname(d)
    return {}


def load_dataset(path: str, tag_map: Dict[str, Dict]) -> Optional[ThermoDataset]:
    with open(path) as f:
        d = json.load(f)
    out = str(d.get("output", ""))
    if out.upper() == "ZPF" or not out:
        return None
    excluded = d.get("excluded_model_contributions")
    tags = d.get("tags") or []
    if excluded is None:
        if tags:
            if all(t in tag_map for t in tags):
                excluded = []
                for t in tags:
                    excluded += list((tag_map[t] or {}).get("excluded_model_contributions", []))
            else:
                excluded = None                      # unresolved
        else:
            excluded = []
    return ThermoDataset(path, out.upper(), [c.upper() for c in d.get("components", [])],
                         [p.upper() for p in d.get("phases", [])], d.get("conditions", {}),
                         np.asarray(d.get("values", []), dtype=float), str(d.get("reference", "")),
                         str(d.get("comment", "")), d.get("solver"), d.get("reference_state"),
                         float(d.get("weight", 1.0)), None if excluded is None else tuple(sorted(set(excluded))),
                         bool(d.get("disabled", False)))


def load_paths(paths: Sequence[str]) -> Tuple[List[ThermoDataset], List[str]]:
    files, out, msgs = [], [], []
    tag_maps: Dict[str, Dict] = {}
    for p in (os.path.abspath(p) for p in paths):
        if os.path.isdir(p):
            fs = sorted(glob.glob(os.path.join(p, "**", "*.json"), recursive=True))
            tag_maps.update({f: find_tag_map(p) for f in fs})
            files += fs
        else:
            files.append(p)
            tag_maps[p] = find_tag_map(os.path.dirname(p))
    for f in files:
        try:
            ds = load_dataset(f, tag_maps.get(f, {}))
        except Exception as ex:
            msgs.append(f"{os.path.basename(f)}: {ex}")
            continue
        if ds is not None:
            out.append(ds)
    return out, msgs


# ---------------------------------------------------------------------------------------------------------
def _points(constituents: List[List[str]], config, occupancies=None) -> Optional[np.ndarray]:
    """Site fractions of a configuration in the order of the phase's (active, sorted) constituents."""
    if occupancies is None:
        occupancies = [None] * len(config)
    pts, i0 = np.zeros(sum(len(s) for s in constituents)), 0
    for subl, cfg, occ in zip(constituents, config, occupancies):
        if isinstance(cfg, (list, tuple)):
            for c, o in zip(cfg, occ):
                if c.upper() not in subl:
                    return None
                pts[i0 + subl.index(c.upper())] = float(o)
        else:
            if cfg.upper() not in subl:
                return None
            pts[i0 + subl.index(cfg.upper())] = 1.0
        i0 += len(subl)
    return pts


class _Props:
    """Compiled property functions of single phases (with model contributions excluded), cached per engine db."""

    def __init__(self, engine):
        self.engine = engine
        cache = getattr(engine, "_thermo_cache", None)
        key = (engine._text_hash, engine.param_names)
        if cache is None or cache[0] != key:
            engine._thermo_cache = (key, {})
        self.cache = engine._thermo_cache[1]

    def get(self, comps, phase: str, output: str, excluded: Tuple[str, ...], values: Dict[str, float]):
        import symengine
        from pycalphad import Model
        from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
        from pycalphad.core.utils import unpack_species
        from pycalphad.model import ReferenceState
        key = (tuple(comps), phase, output, excluded)
        if key not in self.cache:
            dbf = self.engine.dbf
            names = list(self.engine.param_names)
            params = {k: float(values[k]) for k in names}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mod = Model(dbf, list(comps), phase, parameters=params)
                if output.endswith("_FORM"):
                    refs = [ReferenceState(el, dbf.refstates[el]["phase"]) for el in sorted(set(comps) - {"VA"})]
                    mod.shift_reference_state(refs, dbf, contrib_mods={e: symengine.S.Zero for e in excluded})
                    out = output.split("_")[0] + "R"
                else:
                    out = output
                for c in excluded:
                    mod.models[c] = symengine.S.Zero
                    for ref in ("endmember_reference_model", "reference_model"):
                        if hasattr(mod, ref):
                            getattr(mod, ref).models[c] = symengine.S.Zero
                species = sorted(unpack_species(dbf, list(comps)), key=str)
                prf = PhaseRecordFactory(dbf, species, {v.N: 1.0, v.P: 101325.0, v.T: 300.0}, {phase: mod},
                                         parameters=params)
            constituents = [[str(sp.name) for sp in sorted(set(sl) & set(species), key=str)]
                            for sl in mod.constituents]
            self.cache[key] = (mod, prf, out, constituents)
        mod, prf, out, constituents = self.cache[key]
        prf.update_parameters({k: float(values[k]) for k in self.engine.param_names})
        return mod, prf, out, constituents


def residuals(engine, request, values: Dict[str, float], datasets: Sequence[ThermoDataset]):
    """[(dataset path, label, normalised residual)] of the usable data of the request's system, and the number of
    skipped values."""
    from pycalphad import Workspace, calculate
    els = list(request.elements[:2] if request.kind == "binary" else request.elements[:3])
    sysels = frozenset(els)
    data = [d for d in datasets if d.enabled and not d.disabled and d.elements <= sysels and d.elements]
    out, skipped = [], 0
    if not data:
        return out, skipped
    props = _Props(engine)
    for ds in data:
        sig = SIGMA.get(ds.output.split("_")[0][:3] if ds.output.startswith("ACR") else ds.output.split("_")[0])
        if sig is None or ds.excluded is None:
            skipped += ds.values.size
            continue
        sig = sig / max(ds.weight, 1e-12)
        comps = tuple(sorted(ds.elements | {"VA"}))
        Ts = np.atleast_1d(np.asarray(ds.conditions.get("T", 298.15), dtype=float))
        try:
            if ds.kind == "fixed":
                if ds.output.split("_")[-1] not in ("MIX", "FORM"):
                    skipped += ds.values.size
                    continue
                phase = ds.phases[0]
                if phase not in engine.dbf.phases:
                    skipped += ds.values.size
                    continue
                mod, prf, outp, cons = props.get(comps, phase, ds.output, ds.excluded, values)
                cfgs = ds.solver.get("sublattice_configurations", [])
                occs = ds.solver.get("sublattice_occupancies") or [None] * len(cfgs)
                pts = [_points(cons, c, o) for c, o in zip(cfgs, occs)]
                vals = ds.values.reshape(-1, len(Ts), len(cfgs))[0]
                for it, T in enumerate(Ts):
                    good = [i for i, p in enumerate(pts) if p is not None and np.isfinite(vals[it, i])]
                    skipped += len(pts) - len(good)
                    if not good:
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        r = calculate(engine.dbf, list(comps), [phase], output=outp, T=float(T),
                                      P=float(ds.conditions.get("P", 101325)), N=1,
                                      points=np.array([pts[i] for i in good]), model={phase: mod},
                                      phase_records=prf, to_xarray=False)
                    model_vals = np.asarray(getattr(r, outp)).ravel()
                    for k, i in enumerate(good):
                        res = (model_vals[k] - vals[it, i]) / sig
                        out.append((ds.path, f"{ds.label}: {ds.output} {phase} at {T:g} K", float(res)))
            elif ds.kind == "activity":
                el = ds.output.split("_", 1)[1]
                rs = ds.reference_state or {}
                ref_ph = [p for p in rs.get("phases", []) if p in engine.dbf.phases]
                phases = [p for p in ds.phases if p in engine.dbf.phases]
                comps_all = list(comps)
                rc = {v.T: float(rs["conditions"]["T"]), v.P: float(rs["conditions"].get("P", 101325)), v.N: 1.0}
                for k_, val in rs["conditions"].items():
                    if k_.startswith("X_"):
                        rc[v.X(k_[2:])] = float(val)
                rc = {k_: (min(max(x, 1e-12), 1 - 1e-12) if isinstance(k_, v.X) else x) for k_, x in rc.items()}
                # pure-element reference: conditions on the other elements only
                rc = {k_: x for k_, x in rc.items() if not (isinstance(k_, v.X) and str(k_)[2:] == el)} if \
                    len(ds.elements) == 2 else rc
                if len(ds.elements) == 2:
                    other = sorted(ds.elements - {el})[0]
                    rc[v.X(other)] = 1e-12 if any(k_.startswith(f"X_{el}") and float(x) >= 1 - 1e-9
                                                  for k_, x in rs["conditions"].items()) else rc.get(v.X(other), 1e-12)
                from pycalphad.codegen.phase_record_factory import PhaseRecordFactory  # noqa: F401
                models, prf = engine._models(comps, tuple(sorted(set(ref_ph) | set(phases))), values)
                params = {k: float(values[k]) for k in engine.param_names}
                prf.update_parameters(params)
                iel = sorted(ds.elements).index(el)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mu_ref = float(np.asarray(Workspace(engine.dbf, comps_all, ref_ph, rc, models=models,
                                                        phase_record_factory=prf, parameters=params).eq.MU).ravel()[iel])
                    xkeys = [k_ for k_ in ds.conditions if k_.startswith("X_")]
                    xs = {k_[2:]: np.atleast_1d(np.asarray(ds.conditions[k_], float)) for k_ in xkeys}
                    n = len(next(iter(xs.values())))
                    vals = ds.values.reshape(-1, len(Ts), n)[0]
                    for it, T in enumerate(Ts):
                        for i in range(n):
                            a = vals[it, i]
                            if not (a > 0):
                                skipped += 1
                                continue
                            c = {v.T: float(T), v.P: float(ds.conditions.get("P", 101325)), v.N: 1.0}
                            for e_, arr in xs.items():
                                c[v.X(e_)] = float(min(max(arr[i], 1e-9), 1 - 1e-9))
                            mu = float(np.asarray(Workspace(engine.dbf, comps_all, phases, c, models=models,
                                                            phase_record_factory=prf, parameters=params).eq.MU
                                                  ).ravel()[iel])
                            res = ((mu - mu_ref) - R * T * np.log(a)) / sig
                            out.append((ds.path, f"{ds.label}: {ds.output} at {T:g} K", float(res)))
            else:
                skipped += ds.values.size
        except Exception:
            skipped += ds.values.size
    return out, skipped


def thermo_score(engine, request, values: Dict[str, float], datasets: Sequence[ThermoDataset]) -> Optional[ThermoScore]:
    if not any(d.enabled for d in datasets):
        return None
    res, skipped = residuals(engine, request, values, datasets)
    if not res:
        return ThermoScore(0.0, 0, {}, [], skipped)
    per: Dict[str, list] = {}
    for p, _, r in res:
        per.setdefault(p, []).append(r)
    per_ds = {p: (float(np.sqrt(np.mean(np.square(v_)))), len(v_)) for p, v_ in per.items()}
    worst = sorted(((abs(r), w) for _, w, r in res), key=lambda t: -t[0])[:8]
    return ThermoScore(float(np.sqrt(np.mean(np.square([r for _, _, r in res])))), len(res), per_ds, worst, skipped)
