"""Saved diagram views (the buttons above the plot) and the map-result cache."""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

from . import units as U
from .engine import MapRequest

# settings that define *which* diagram a view is (zoom limits are stored separately)
_FLOAT_KEYS = ("tmin_K", "tmax_K", "tstep", "tsec_K", "xmin", "xmax", "xstep", "ymin", "ymax", "fixed")


@dataclass
class SavedView:
    name: str
    settings: Dict
    limits: Optional[List[List[float]]] = None      # [[x0, x1], [y0, y1]] in the view's display units

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "SavedView":
        return cls(d["name"], dict(d["settings"]), d.get("limits"))


def same_diagram(a: Dict, b: Dict) -> bool:
    """True when two diagram settings describe the same diagram (floats compared with a tolerance)."""
    for k in ("kind", "t_unit", "x_unit", "cg", "solver", "pdens"):
        if a.get(k) != b.get(k):
            return False
    n = 2 if a.get("kind") == "binary" else 3
    if list(a.get("elements", []))[:n] != list(b.get("elements", []))[:n]:
        return False
    relevant = {"binary": ("tmin_K", "tmax_K", "tstep", "xmin", "xmax", "xstep"),
                "ternary": ("tsec_K", "xmin", "xmax", "ymin", "ymax", "xstep"),
                "isopleth": ("tmin_K", "tmax_K", "tstep", "xmin", "xmax", "xstep", "fixed")}[a.get("kind", "binary")]
    for k in relevant:
        x, y = float(a.get(k, 0)), float(b.get(k, 0))
        if abs(x - y) > 1e-6 * max(1.0, abs(x), abs(y)) + 5e-3:
            return False
    return True


def auto_name(g: Dict, zoomed: bool = False) -> str:
    els = "-".join(e.capitalize() for e in g["elements"][: 2 if g["kind"] == "binary" else 3])
    tu = g.get("t_unit", "K")

    def T(k):
        return f"{float(U.t_from_K(g[k], tu)):.6g}"

    if g["kind"] == "ternary":
        s = f"{els} {T('tsec_K')} {tu}"
        if g.get("xmin", 0) > 0 or g.get("xmax", 100) < 100 or g.get("ymin", 0) > 0 or g.get("ymax", 100) < 100:
            s += " (part)"
    elif g["kind"] == "isopleth":
        s = f"{els} x({g['elements'][2].capitalize()})={g['fixed']:g}%"
    else:
        s = f"{els} {T('tmin_K')}–{T('tmax_K')} {tu}"
        if g.get("xmin", 0) > 0 or g.get("xmax", 100) < 100:
            s += f", {g['xmin']:g}–{g['xmax']:g}%"
    return s + (" (zoom)" if zoomed else "")


def request_key(req: MapRequest) -> Tuple:
    """Hashable identity of a map request, robust to float round-off from unit conversions."""
    out = []
    for k, v in sorted(asdict(req).items()):
        if isinstance(v, float):
            v = round(v, 6)
        elif isinstance(v, list):
            v = tuple(v)
        out.append((k, v))
    return tuple(out)


def values_stamp(compute_text: str, values: Dict[str, float]) -> Tuple:
    return (hashlib.sha1(compute_text.encode()).hexdigest(), tuple(sorted(values.items())))


class ResultCache:
    """Last results keyed by (request, database structure + parameter values)."""

    def __init__(self, size: int = 40):
        self.size = size
        self._d: "OrderedDict[Tuple, object]" = OrderedDict()

    def get(self, req_key: Tuple, stamp: Tuple):
        k = (req_key, stamp)
        if k in self._d:
            self._d.move_to_end(k)
            return self._d[k]
        return None

    def put(self, req_key: Tuple, stamp: Tuple, result) -> None:
        self._d[(req_key, stamp)] = result
        self._d.move_to_end((req_key, stamp))
        while len(self._d) > self.size:
            self._d.popitem(last=False)

    def latest_for(self, req_key: Tuple):
        """Most recent result for a request with any parameter values (shown while the up-to-date map runs)."""
        for (rk, _), res in reversed(self._d.items()):
            if rk == req_key:
                return res
        return None

    def clear(self):
        self._d.clear()
