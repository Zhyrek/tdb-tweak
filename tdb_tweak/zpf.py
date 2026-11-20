"""ESPEI ZPF datasets (``"output": "ZPF"``) for plotting over a computed diagram."""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class ZPFVertex:
    phase: str
    fractions: Dict[str, float]          # element -> mole fraction for the elements given (others implied)
    known: bool                          # False when ESPEI has null compositions (phase present, comp unknown)


@dataclass
class ZPFTieline:
    T: float
    vertices: List[ZPFVertex]


@dataclass
class ZPFDataset:
    path: str
    components: List[str]
    phases: List[str]
    reference: str
    comment: str
    tielines: List[ZPFTieline] = field(default_factory=list)
    enabled: bool = True

    @property
    def elements(self) -> frozenset:
        return frozenset(c.upper() for c in self.components if c.upper() != "VA")

    @property
    def label(self) -> str:
        return f"{self.reference or os.path.basename(self.path)}"


def load_dataset(path: str) -> Optional[ZPFDataset]:
    with open(path) as f:
        d = json.load(f)
    if str(d.get("output", "")).upper() != "ZPF":
        return None
    T = d.get("conditions", {}).get("T")
    values = d.get("values", [])
    Ts = np.broadcast_to(np.atleast_1d(np.asarray(T, dtype=float)), (len(values),)) if len(values) else []
    ds = ZPFDataset(path, [c.upper() for c in d.get("components", [])], [p.upper() for p in d.get("phases", [])],
                    str(d.get("reference", "")), str(d.get("comment", "")))
    for Ti, tl in zip(Ts, values):
        verts = []
        for entry in tl:
            phase, comps, fracs = entry[0], entry[1], entry[2]
            known = fracs is not None and all(f is not None for f in fracs)
            fd = {c.upper(): float(f) for c, f in zip(comps, fracs)} if known else {}
            verts.append(ZPFVertex(str(phase).upper(), fd, known))
        ds.tielines.append(ZPFTieline(float(Ti), verts))
    return ds


def load_paths(paths: Sequence[str]) -> Tuple[List[ZPFDataset], List[str]]:
    """Files and/or folders (searched recursively for *.json). Returns (datasets, messages)."""
    files: List[str] = []
    for p in (os.path.abspath(p) for p in paths):
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "**", "*.json"), recursive=True))
        else:
            files.append(p)
    out, msgs = [], []
    for f in files:
        try:
            ds = load_dataset(f)
        except Exception as ex:
            msgs.append(f"{os.path.basename(f)}: {ex}")
            continue
        if ds is not None:
            out.append(ds)
    return out, msgs


def fractions_for(v: ZPFVertex, elements: Sequence[str]) -> Optional[Dict[str, float]]:
    """Mole fraction of every element of ``elements`` (the dependent one by difference)."""
    if not v.known:
        return None
    given = {e: v.fractions[e] for e in elements if e in v.fractions}
    rest = [e for e in elements if e not in given]
    if len(rest) > 1:
        return None
    if rest:
        given[rest[0]] = 1.0 - sum(given.values())
    return given


def matching(datasets: Sequence[ZPFDataset], elements: Sequence[str]) -> List[ZPFDataset]:
    els = frozenset(e.upper() for e in elements)
    return [d for d in datasets if d.elements == els]


def temperatures(datasets: Sequence[ZPFDataset], elements: Sequence[str], decimals: int = 1) -> List[Tuple[float, int]]:
    """Distinct temperatures with the number of tie-lines at each (for choosing an isothermal section)."""
    counts: Dict[float, int] = {}
    for d in matching(datasets, elements):
        if not d.enabled:
            continue
        for tl in d.tielines:
            T = round(tl.T, decimals)
            counts[T] = counts.get(T, 0) + 1
    return sorted(counts.items())
