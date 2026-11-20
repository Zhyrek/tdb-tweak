"""Drawing a computed map, literature images and ZPF data on one matplotlib Axes (no Qt here)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
from matplotlib.collections import LineCollection

from . import units as U
from .engine import MapResult
from .zpf import ZPFDataset, fractions_for, matching

_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b", "#e377c2",
            "#bcbd22", "#7f7f7f", "#393b79", "#ad494a", "#637939", "#7b4173", "#e7ba52", "#3182bd",
            "#843c39", "#31a354", "#756bb1", "#636363"]


class PhaseColors:
    """Stable phase -> colour (a phase keeps its colour across redraws)."""

    def __init__(self):
        self._map: Dict[str, str] = {}

    @staticmethod
    def base(phase: str) -> str:
        return re.sub(r"#\d+$", "", phase)

    def __call__(self, phase: str) -> str:
        b = self.base(phase)
        if b not in self._map:
            self._map[b] = _PALETTE[len(self._map) % len(_PALETTE)]
        return self._map[b]


@dataclass
class View:
    kind: str                                 # binary / ternary / isopleth
    elements: Tuple[str, ...]
    t_unit: str = "K"
    x_unit: str = "at.%"
    masses: Dict[str, float] = field(default_factory=dict)
    temperature: float = 1000.0               # ternary section (K)
    fixed: float = 0.1                        # isopleth X(C)
    zpf_tolerance: float = 5.0                # K, ternary
    tieline_every: int = 0                    # 0: none; binary default off, ternary 3
    show_invariants: bool = True
    zpf_misfit: Optional[Dict] = None         # (path, tie-line index) -> largest |driving force| (J/mol)


# ---------------------------------------------------------------------------------------------------------
# coordinate conversion
# ---------------------------------------------------------------------------------------------------------
def to_plot(view: View, x, y):
    """Map-result coordinates (mole fractions, K) -> plot coordinates."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if view.kind == "binary":
        A, B = view.elements
        return U.binary_x_to_display(x, A, B, view.masses, view.x_unit), U.t_from_K(y, view.t_unit)
    if view.kind == "ternary":
        A, B, C = view.elements
        fr = {A: 1 - x - y, B: x, C: y}
        if U.is_mass(view.x_unit):
            d = U.x_to_display(fr, [A, B, C], view.masses, "mass fraction", [B, C])
            x, y = d[B], d[C]
        return U.tri_to_xy(x, y)
    # isopleth: x = X(B) at fixed X(C)
    A, B, C = view.elements
    fr = {A: 1 - x - view.fixed, B: x, C: np.full_like(x, view.fixed)}
    return U.x_to_display(fr, [A, B, C], view.masses, view.x_unit, [B])[B], U.t_from_K(y, view.t_unit)


def axis_labels(view: View) -> Tuple[str, str]:
    comp = {"at.%": "at.%", "mole fraction": "mole fraction", "wt.%": "wt.%", "mass fraction": "mass fraction"}
    if view.kind in ("binary", "isopleth"):
        B = view.elements[1].capitalize()
        return f"{B} ({comp[view.x_unit]})", f"Temperature ({view.t_unit})"
    return "", ""


# ---------------------------------------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------------------------------------
def draw_ternary_frame(ax, view: View, grid: float = 0.1):
    A, B, C = (e.capitalize() for e in view.elements)
    tri = np.array([[0, 0], [1, 0], [0.5, U.SQ3], [0, 0]])
    ax.plot(tri[:, 0], tri[:, 1], color="black", lw=1.0, zorder=1.5)
    segs = []
    for g in np.arange(grid, 1.0 - 1e-9, grid):
        # lines of constant B, C and A
        for (b0, c0), (b1, c1) in (((g, 0), (g, 1 - g)), ((0, g), (1 - g, g)), ((1 - g, 0), (0, 1 - g))):
            x0, y0 = U.tri_to_xy(b0, c0)
            x1, y1 = U.tri_to_xy(b1, c1)
            segs.append([(float(x0), float(y0)), (float(x1), float(y1))])
    ax.add_collection(LineCollection(segs, colors="#d0d0d0", linewidths=0.5, zorder=0.5))
    unit = "wt." if U.is_mass(view.x_unit) else ""
    ax.text(-0.03, -0.03, A, ha="right", va="top", fontsize=11)
    ax.text(1.03, -0.03, B, ha="left", va="top", fontsize=11)
    ax.text(0.5, U.SQ3 + 0.03, C, ha="center", va="bottom", fontsize=11)
    for g in np.arange(0.2, 1.0 - 1e-9, 0.2):
        x, y = U.tri_to_xy(g, 0)
        ax.text(float(x), -0.025, f"{g * 100:.0f}", ha="center", va="top", fontsize=7, color="#555")
        x, y = U.tri_to_xy(0, g)
        ax.text(float(x) - 0.02, float(y), f"{g * 100:.0f}", ha="right", va="center", fontsize=7, color="#555")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{A}-{B}-{C}  {U.t_from_K(view.temperature, view.t_unit):.6g} {view.t_unit}"
                 + (f"  ({unit}%)" if unit else ""), fontsize=10)


def draw_result(ax, view: View, res: Optional[MapResult], colors: PhaseColors, linewidth: float = 1.4):
    handles = {}
    if res is None:
        return handles
    for b in res.boundaries:
        if len(b.x) == 0:
            continue
        X, Y = to_plot(view, b.x, b.y)
        (ln,) = ax.plot(X, Y, color=colors(b.phase), lw=linewidth, zorder=3, solid_capstyle="butt")
        handles.setdefault(colors.base(b.phase), ln)
        if len(b.x) == 1 or (np.ptp(X) == 0 and np.ptp(Y) == 0):
            ax.scatter([X[0]], [Y[0]], color=colors(b.phase), s=10, zorder=3)
    if view.tieline_every and res.tielines:
        segs = []
        for tl in res.tielines:
            for i in range(0, tl.shape[0], view.tieline_every):
                (x0, y0), (x1, y1) = tl[i]
                X, Y = to_plot(view, [x0, x1], [y0, y1])
                segs.append(list(zip(X, Y)))
        ax.add_collection(LineCollection(segs, colors="#7fbf7f", linewidths=0.5, zorder=2))
    if view.show_invariants:
        for names, xs, ys in res.invariants:
            X, Y = to_plot(view, xs, ys)
            if view.kind == "binary":
                ax.plot([np.min(X), np.max(X)], [np.mean(Y)] * 2, color="black", lw=1.2, zorder=3.5)
            elif view.kind == "ternary":
                ax.fill(np.r_[X, X[:1]], np.r_[Y, Y[:1]], facecolor="#f2c4c4", edgecolor="#c03030", lw=0.8,
                        alpha=0.6, zorder=2.5)
            else:
                ax.plot(X, Y, color="black", lw=1.2, zorder=3.5)
    return handles


def draw_zpf(ax, view: View, datasets: Sequence[ZPFDataset], colors: PhaseColors) -> List:
    """Returns pickable artists; each has ``.zpf_info`` (list of strings, one per point)."""
    artists = []
    els = list(view.elements) if view.kind != "isopleth" else []
    if not els:
        return artists
    for ds in matching(datasets, els):
        if not ds.enabled:
            continue
        pts_by_phase: Dict[str, List] = {}
        segs = []
        for i_tl, tl in enumerate(ds.tielines):
            if view.kind == "ternary" and abs(tl.T - view.temperature) > view.zpf_tolerance:
                continue
            verts = []
            for i_v, vx in enumerate(tl.vertices):
                fr = fractions_for(vx, els)
                if fr is None:
                    continue
                if view.kind == "binary":
                    X, Y = to_plot(view, fr[els[1]], tl.T)
                else:
                    X, Y = to_plot(view, fr[els[1]], fr[els[2]])
                X, Y = float(X), float(Y)
                verts.append((X, Y))
                info = f"{ds.label}: {vx.phase} at {U.t_from_K(tl.T, view.t_unit):.6g} {view.t_unit}"
                # a tie-line's misfit often sits on a vertex that is not drawn (unknown composition), so each point
                # shows its tie-line's largest driving force
                df = view.zpf_misfit.get((ds.path, i_tl)) if view.zpf_misfit is not None else None
                if df is not None:
                    info += f" — tie-line misfit {df:.0f} J/mol"
                pts_by_phase.setdefault(vx.phase, []).append((X, Y, info, tl.T, df))
            if len(verts) >= 2:
                if len(verts) == 2:
                    segs.append(verts)
                else:
                    segs.append(verts + [verts[0]])
        if segs:
            ax.add_collection(LineCollection(segs, colors="#555555", linewidths=0.7, linestyles="--", zorder=4))
        for phase, pts in pts_by_phase.items():
            if view.zpf_misfit is not None:
                # fill: the phase; outline: how badly the point fits (green 0 → red ≥ 1000 J/mol, ESPEI's ZPF
                # uncertainty, and thicker)
                bad = np.array([min(abs(p[4]) / 1000.0, 1.0) if p[4] is not None else np.nan for p in pts])
                cmap = matplotlib.colormaps["RdYlGn_r"]
                edges = [cmap(0.1 + 0.9 * b) if np.isfinite(b) else (0, 0, 0, 1) for b in bad]
                widths = [1.2 + 2.6 * b if np.isfinite(b) else 0.6 for b in bad]
                sizes = [34 + 40 * b if np.isfinite(b) else 26 for b in bad]
            else:
                edges, widths, sizes = "black", 0.6, 26
            sc = ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=sizes, marker="o",
                            facecolor=colors(phase), edgecolor=edges, linewidths=widths, zorder=5, picker=4)
            sc.zpf_info = [p[2] for p in pts]
            sc.zpf_T = [p[3] for p in pts]
            artists.append(sc)
    return artists


def draw_images(ax, images: Sequence[Tuple[np.ndarray, Tuple[float, float, float, float], float]]):
    for arr, extent, alpha in images:
        ax.imshow(arr, extent=extent, origin="lower", alpha=alpha, zorder=0, aspect="auto",
                  interpolation="bilinear")


def draw_ghost(ax, view: View, res: Optional[MapResult]):
    """The map of the checkpoint parameters, faint and grey, behind the current one."""
    if res is None:
        return
    for b in res.boundaries:
        if len(b.x):
            X, Y = to_plot(view, b.x, b.y)
            ax.plot(X, Y, color="#9a9a9a", lw=3.2, alpha=0.45, zorder=1.5, solid_capstyle="round")
    for names, xs, ys in res.invariants:
        X, Y = to_plot(view, xs, ys)
        if view.kind == "binary":
            ax.plot([np.min(X), np.max(X)], [np.mean(Y)] * 2, color="#9a9a9a", lw=3.0, alpha=0.45, zorder=1.5)
        elif view.kind == "ternary":
            ax.plot(np.r_[X, X[:1]], np.r_[Y, Y[:1]], color="#9a9a9a", lw=2.0, alpha=0.45, zorder=1.5)


def draw_all(fig, view: View, res: Optional[MapResult], datasets: Sequence[ZPFDataset],
             images: Sequence, colors: PhaseColors, limits: Optional[Tuple] = None, ghost: Optional[MapResult] = None):
    """Clear ``fig`` and draw everything. Returns (ax, pickable zpf artists)."""
    fig.clear()
    ax = fig.add_subplot(111)
    draw_images(ax, images)
    draw_ghost(ax, view, ghost)
    handles = draw_result(ax, view, res, colors)
    zpf_artists = draw_zpf(ax, view, datasets, colors)
    if view.kind == "ternary":
        draw_ternary_frame(ax, view)
        if limits is None:
            ax.set_xlim(-0.06, 1.06)
            ax.set_ylim(-0.08, U.SQ3 + 0.08)
    else:
        xl, yl = axis_labels(view)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        if limits is None and res is not None and res.request is not None:
            r = res.request
            if view.kind == "binary":
                A, B = view.elements
                x0 = U.binary_x_to_display(r.xmin, A, B, view.masses, view.x_unit)
                x1 = U.binary_x_to_display(r.xmax, A, B, view.masses, view.x_unit)
            else:
                x0, _ = to_plot(view, np.array([r.xmin]), np.array([r.tmin]))
                x1, _ = to_plot(view, np.array([r.xmax]), np.array([r.tmax]))
                x0, x1 = float(x0[0]), float(x1[0])
            ax.set_xlim(float(x0), float(x1))
            ax.set_ylim(float(U.t_from_K(r.tmin, view.t_unit)), float(U.t_from_K(r.tmax, view.t_unit)))
        title = "-".join(e.capitalize() for e in view.elements)
        if view.kind == "isopleth":
            title += f"   x({view.elements[2].capitalize()}) = {view.fixed:g}"
        ax.set_title(title, fontsize=10)
    if limits is not None:
        ax.set_xlim(*limits[0])
        ax.set_ylim(*limits[1])
    if handles:
        names = sorted(handles)
        ax.legend([handles[n] for n in names], names, loc="center left", bbox_to_anchor=(1.01, 0.5),
                  fontsize=8, frameon=False)
    fig.subplots_adjust(left=0.09, right=0.80, bottom=0.09, top=0.94)
    return ax, zpf_artists
