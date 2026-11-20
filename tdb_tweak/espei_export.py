"""ESPEI ZPF datasets from optimiser targets and from points marked by hand (first pass).

ESPEI's ZPF format: ``values`` is a list of tie-lines; each tie-line lists the phases in equilibrium with
``[phase, [elements], [mole fractions]]`` (``null`` fractions: the phase is present, its composition unknown);
``conditions.T`` gives each tie-line's temperature (``broadcast_conditions: false``)."""
from __future__ import annotations

import json
from typing import Dict, List, Sequence, Tuple

from .zpf import ZPFDataset, ZPFTieline, ZPFVertex


def _vertex(phase: str, indep: Sequence[str], fr: Dict[str, float] = None) -> list:
    if fr is None:
        return [phase, list(indep), [None] * len(indep)]
    return [phase, list(indep), [round(float(fr[e]), 6) for e in indep]]


def tielines_from_entry(e: Dict) -> List[Tuple[float, list]]:
    """(T, tie-line) of an optimiser target entry, at its *wanted* position."""
    els = list(e["elements"])
    indep = els[1:]
    T = float(e.get("want_T", e["T"]))
    names = list(e["names"])
    cur, want = e["cur_x"], e["want_x"]
    if e["kind"] == "invariant":
        return [(T, [_vertex(n, indep, {indep[0]: want.get(n, cur[n])}) for n in names])]
    if e["kind"] == "congruent":
        x = want.get(names[0], cur[names[0]])
        return [(T, [_vertex(n, indep, {indep[0]: x}) for n in names])]
    if e["kind"] == "boundary" and len(els) == 2:
        return [(T, [_vertex(names[0], indep, {indep[0]: want[names[0]]}), _vertex(names[1], indep)])]
    if e["kind"] == "boundary":
        fr = {indep[0]: want[f"node|{indep[0]}"], indep[1]: want[f"node|{indep[1]}"]}
        return [(float(e["T"]), [_vertex(names[0], indep, fr), _vertex(names[1], indep)])]
    if e["kind"] == "tietriangle":
        verts = []
        for n in names:
            fr = {el: want.get(f"{n}|{el}", cur[f"{n}|{el}"]) for el in indep}
            verts.append(_vertex(n, indep, fr))
        return [(float(e["T"]), verts)]
    return []


def dataset_dicts(entries: Sequence[Dict], reference: str = "tdb-tweak targets") -> Dict[tuple, dict]:
    """One ESPEI ZPF dataset per system: {elements: dataset dict}."""
    out: Dict[tuple, dict] = {}
    for e in entries:
        els = tuple(sorted(e["elements"]))
        d = out.setdefault(els, {"components": list(els), "phases": [], "conditions": {"P": 101325, "T": []},
                                 "broadcast_conditions": False, "output": "ZPF", "values": [],
                                 "reference": reference,
                                 "comment": "Targets set by hand in tdb-tweak (positions where the features should be)"})
        for T, tl in tielines_from_entry(e):
            d["conditions"]["T"].append(round(T, 4))
            d["values"].append(tl)
            for v in tl:
                if v[0] not in d["phases"]:
                    d["phases"].append(v[0])
    return out


def marked_to_dict(ds: ZPFDataset) -> dict:
    els = sorted(ds.elements)
    indep = els[1:]
    phases, values, Ts = [], [], []
    for tl in ds.tielines:
        Ts.append(round(tl.T, 4))
        row = []
        for v in tl.vertices:
            row.append(_vertex(v.phase, indep, v.fractions if v.known else None))
            if v.phase not in phases:
                phases.append(v.phase)
        values.append(row)
    return {"components": els, "phases": phases, "conditions": {"P": 101325, "T": Ts},
            "broadcast_conditions": False, "output": "ZPF", "values": values,
            "reference": ds.reference or "marked by hand", "comment": ds.comment}


def write(path: str, d: dict) -> str:
    with open(path, "w") as f:
        json.dump(d, f, indent=1)
    return path


def add_mark(ds: ZPFDataset, T: float, phase: str, fractions: Dict[str, float], partner: str) -> None:
    ds.tielines.append(ZPFTieline(float(T), [ZPFVertex(phase, dict(fractions), True), ZPFVertex(partner, {}, False)]))
    for p in (phase, partner):
        if p not in ds.phases:
            ds.phases.append(p)
