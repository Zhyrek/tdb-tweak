"""Glue between the Optimize panel, the plot (pick + drag targets) and the worker.

Targets
-------
* binary three-phase invariant: a dashed line with one handle per phase;
* binary congruent point: one handle;
* ternary tie triangle (isothermal section): a dashed triangle with one handle per corner.

A phase's handle only moves within the compositions its sublattice model can reach: not at all for a line compound,
along a line for a "1-D stoichiometric" phase such as (Fe)(Ni,Cr), freely otherwise (points.composition_space).
Several targets can be active; all of them are optimised together."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QMenu

from . import units as U
from .diagram import to_plot
from .optimize import Variable, linear_step, target_label
from .optpanel import OptPanel, base_phase, entry_label, fixed_phases, new_entry
from .paramtable import apply_symbol_values, selected_rows, visible_rows
from .points import constrain

PICK_PX = 25      # right-click distance to an invariant line / congruent point / tie triangle
GRAB_PX = 10      # press distance to a handle / a target line
COLOR = "#e0119d"


def _signature(e: Dict) -> tuple:
    return (e["kind"], tuple(sorted(e["names"])))


def _distance(a: Dict, b: Dict) -> float:
    if a["kind"] == "boundary" and a.get("direction") is not None:
        n = a["names"][0]
        return sum(abs(a["cur_x"][k] - b["cur_x"].get(k, 9.0)) for k in a["cur_x"] if k.startswith(n + "|"))
    if a["kind"] == "boundary":
        return abs(a["T"] - b["T"]) / 100.0 + abs(a["cur_x"][a["names"][0]] - b["cur_x"][b["names"][0]])
    if a["kind"] == "tietriangle":
        return sum(abs(a["cur_x"][k] - b["cur_x"].get(k, 9.0)) for k in a["cur_x"])
    return abs(a["T"] - b["T"]) / 100.0


class OptController:
    def __init__(self, win):
        self.win = win
        self.panel = OptPanel()
        self.entries: List[Dict] = []
        self.visible = True
        self._drag = None                   # (entry index, "line" / "point" / key)
        self._artists: List[Dict] = []
        self._lin = None                    # live drag: {"key": ..., "linear": ...} (valid for one parameter set)
        self._lin_key = None
        self._live = None                   # an active live drag: {"start": {symbol: value}, "applied": bool}
        self._pred: Dict = {}               # live drag: predicted target quantities {(entry index, key): value}
        self._items: List[Dict] = []
        p = self.panel
        p.addRequested.connect(lambda i: self.add(self._items[i]) if 0 <= i < len(self._items) else None)
        p.removeRequested.connect(self.remove)
        p.clearRequested.connect(self.clear)
        p.export_btn.clicked.connect(self.export_targets)
        p.targetsEdited.connect(self.update_overlay)
        p.runRequested.connect(self.run)
        p.stopRequested.connect(self.win.runner.cancel)
        p.applyIteration.connect(self.apply_iteration)
        p.useVisibleRequested.connect(lambda: self.fill_variables(selection=False))
        p.useSelectionRequested.connect(lambda: self.fill_variables(selection=True))
        p.phaseLocksChanged.connect(lambda: self.win.params.model.refresh_values())
        win.params.locksChanged.connect(p.refresh_locks)
        win.runner.opt_progress.connect(self._on_progress)
        win.runner.opt_finished.connect(self._on_finished)
        c = win.canvas
        c.mpl_connect("button_press_event", self._on_press)
        c.mpl_connect("motion_notify_event", self._on_motion)
        c.mpl_connect("button_release_event", self._on_release)

    # ------------------------------------------------------------------------------------------ points
    def _points(self) -> List[Dict]:
        """Optimisable features of the map on screen."""
        w = self.win
        view = w.current_view()
        if view.kind not in ("binary", "ternary") or w.doc is None:
            return []
        res = w._result_for_display(view)
        if res is None:
            return []
        spaces = getattr(res, "phase_space", {}) or {}
        els = list(view.elements)
        out = []
        if view.kind == "binary":
            for n, xs, ys in res.invariants:
                out.append(dict(kind="invariant", names=list(n), T=float(np.mean(ys)), spaces=spaces, elements=els,
                                cur_x={p: float(x) for p, x in zip(n, xs)}))
            for c in getattr(res, "congruent", []):
                out.append(dict(kind="congruent", names=list(c.phases), T=float(c.T), spaces=spaces, elements=els,
                                cur_x={c.phases[0]: float(c.x)}, fixed_x=bool(c.fixed_x), compound=c.compound))
        else:
            T = float(res.request.temperature)
            for n, xb, xc in res.invariants:
                cur = {}
                for p, b, c in zip(n, xb, xc):
                    cur[f"{p}|{els[1]}"] = float(b)
                    cur[f"{p}|{els[2]}"] = float(c)
                out.append(dict(kind="tietriangle", names=list(n), T=T, spaces=spaces, elements=els, cur_x=cur))
        return out

    def _boundary_curves(self):
        w = self.win
        view = w.current_view()
        res = w._result_for_display(view) if view.kind in ("binary", "ternary") else None
        return [] if res is None else [b for b in res.boundaries if b.partner and len(b.x) > 1]

    def _boundary_x(self, e: Dict, T: float) -> Optional[Dict]:
        """Boundary compositions of an entry's two phases at T on the current map (nearest branch)."""
        best = None
        for b in self._boundary_curves():
            if b.phase != e["names"][0] or b.partner != e["names"][1]:
                continue
            for k in range(len(b.y) - 1):
                y0, y1 = b.y[k], b.y[k + 1]
                if (y0 - T) * (y1 - T) > 0 or y0 == y1:
                    continue
                f = (T - y0) / (y1 - y0)
                x = b.x[k] + f * (b.x[k + 1] - b.x[k])
                xp = b.partner_x[k] + f * (b.partner_x[k + 1] - b.partner_x[k])
                d = abs(x - e["cur_x"][e["names"][0]])
                if best is None or d < best[0]:
                    best = (d, {e["names"][0]: float(x), e["names"][1]: float(xp)})
        return None if best is None else best[1]

    def _refresh_ternary_node(self, e: Dict):
        """Current state of a ternary boundary node: the nearest point of its boundary on the map."""
        els = e["elements"]
        a, b = e["names"]
        p = np.array([e["want_x"][f"node|{els[1]}"], e["want_x"][f"node|{els[2]}"]])
        best = None
        for bb in self._boundary_curves():
            if bb.phase != a or bb.partner != b or bb.partner_y is None:
                continue
            P = np.c_[bb.x, bb.y]
            Q = np.c_[bb.partner_x, bb.partner_y]
            for k in range(len(P) - 1):                 # the nearest point on the curve (segments interpolated)
                sv = P[k + 1] - P[k]
                t = float(np.clip(np.dot(p - P[k], sv) / max(float(np.dot(sv, sv)), 1e-15), 0.0, 1.0))
                q = P[k] + t * sv
                dd = float(np.linalg.norm(q - p))
                if best is None or dd < best[0]:
                    best = (dd, q, Q[k] + t * (Q[k + 1] - Q[k]))
        if best is None:
            return
        _, pa, pb = best
        e["cur_x"].update({f"{a}|{els[1]}": float(pa[0]), f"{a}|{els[2]}": float(pa[1]),
                           f"{b}|{els[1]}": float(pb[0]), f"{b}|{els[2]}": float(pb[1]),
                           f"node|{els[1]}": float(pa[0]), f"node|{els[2]}": float(pa[1])})
        d = pb - pa
        if np.linalg.norm(d) > 1e-9:
            e["direction"] = (d / np.linalg.norm(d)).tolist()

    def refresh_invariants(self):
        """After a map: refresh the 'points on this map' list and the current values of the targets."""
        self._items = self._points()
        view = self.win.current_view()
        for e in self.entries:
            if e["kind"] == "boundary" and e.get("direction") is not None:
                if view.kind == "ternary" and list(e["elements"]) == list(view.elements):
                    self._refresh_ternary_node(e)
                continue
            if e["kind"] == "boundary":
                if view.kind == "binary" and list(e["elements"]) == list(view.elements):
                    got = self._boundary_x(e, e["want_T"])
                    if got is not None:
                        e["cur_x"], e["T"] = got, e["want_T"]
                continue
            cands = [it for it in self._items if _signature(it) == _signature(e)]
            if cands:
                it = min(cands, key=lambda c: _distance(e, c))
                e["T"], e["cur_x"], e["names"], e["spaces"] = it["T"], it["cur_x"], it["names"], it["spaces"]
        self.panel.set_points(self._items, self.win.t_unit.currentText())
        self.panel.refresh_values()

    def _index_of(self, item: Dict) -> Optional[int]:
        for i, e in enumerate(self.entries):
            if _signature(e) == _signature(item) and _distance(e, item) < 0.05:
                return i
        return None

    def add(self, item: Dict):
        item = dict(item)
        item.setdefault("req", self.win.build_request())
        if self._index_of(item) is None:
            self.entries.append(new_entry(item))
        self._sync_panel()
        if not self.panel._rows:
            self.fill_variables(selection=False)
        self.visible = True
        self.win.tabs.setCurrentWidget(self.panel)
        self.win.replot()

    def remove(self, i: int):
        if 0 <= i < len(self.entries):
            del self.entries[i]
            self._sync_panel()
            self.win.replot()

    def clear(self):
        self.entries = []
        self._sync_panel()
        self.win.replot()

    def _sync_panel(self):
        self.panel.set_entries(self.entries, self.win.t_unit.currentText())
        self.panel.clear_history()

    def fill_variables(self, selection: bool):
        w = self.win
        if w.doc is None:
            return
        rows = selected_rows(w.params) if selection else [r for r in visible_rows(w.params) if r.kind == "VV"]
        seen, uniq = set(), []
        for r in rows:
            if r.symbol not in seen:
                seen.add(r.symbol)
                uniq.append(r)
        if selection and not uniq:
            w.statusBar().showMessage("Select rows in the parameter table first.", 5000)
            return
        # chosen plain coefficients become parameters now, so the maps the targets are picked from use the same
        # compiled models as the optimiser (pycalphad's mapping can differ between the two compilations)
        new_live = [r for r in uniq if r.kind == "COEF" and r.rid not in w.doc.live]
        for r in new_live:
            w.doc.live.add(r.rid)
        if new_live:
            w._log(f"{len(new_live)} coefficients became parameters; mapping again with them")
            w.redraw()
        phases_of: Dict[str, set] = {}
        for r in w.doc.rows:
            if r.symbol in seen:
                phases_of.setdefault(r.symbol, set()).add(r.phase or "(functions)")
        self.panel.set_variable_rows(uniq, {r.symbol: w.doc.value(r) for r in uniq}, phases_of, doc=w.doc)

    # ------------------------------------------------------------------------------------------ constraints
    def _constrain(self, e: Dict):
        """Keep wanted compositions inside what each phase can reach (1-D stoichiometric phases)."""
        if e["kind"] != "tietriangle":
            return
        B, C = e["elements"][1], e["elements"][2]
        for n in e["names"]:
            kb, kc = f"{n}|{B}", f"{n}|{C}"
            if kb not in e["want_x"]:
                continue
            sp = (e.get("spaces") or {}).get(base_phase(n))
            b, c = e["want_x"][kb], e["want_x"][kc]
            if sp is not None and sp["rank"] == 1:
                b, c = constrain(sp, [b, c])
            if b + c > 1:
                s = b + c
                b, c = b / s, c / s
            e["want_x"][kb], e["want_x"][kc] = float(np.clip(b, 0, 1)), float(np.clip(c, 0, 1))

    # ------------------------------------------------------------------------------------------ overlay
    def _view(self):
        return self.win.current_view()

    def _xy(self, e: Dict):
        """Plot coordinates of the wanted position of an entry: (X, Y, handle keys, movable flags)."""
        view = self._view()
        if e["kind"] == "boundary" and e.get("direction") is not None:
            els = e["elements"]
            X, Y = to_plot(view, np.array([e["want_x"][f"node|{els[1]}"]]), np.array([e["want_x"][f"node|{els[2]}"]]))
            return X, Y, ["point"], [True]
        if e["kind"] == "boundary":
            X, Y = to_plot(view, np.array([e["want_x"][e["names"][0]]]), np.array([e["want_T"]]))
            return X, Y, ["point"], [True]
        if e["kind"] == "congruent":
            x = e["want_x"].get(e["names"][0], e["cur_x"][e["names"][0]])
            X, Y = to_plot(view, np.array([x]), np.array([e["want_T"]]))
            return X, Y, ["point"], [True]
        fixed = set(fixed_phases(e))
        if e["kind"] == "invariant":
            names = list(e["names"])
            xs = [e["want_x"].get(n, e["cur_x"][n]) for n in names]
            X, Y = to_plot(view, np.array(xs), np.full(len(names), e["want_T"]))
            return X, Y, names, [n not in fixed for n in names]
        B, C = e["elements"][1], e["elements"][2]
        names = list(e["names"])
        bs = [e["want_x"].get(f"{n}|{B}", e["cur_x"][f"{n}|{B}"]) for n in names]
        cs = [e["want_x"].get(f"{n}|{C}", e["cur_x"][f"{n}|{C}"]) for n in names]
        X, Y = to_plot(view, np.array(bs), np.array(cs))
        return X, Y, names, [n not in fixed for n in names]

    def draw_overlay(self, ax):
        self._artists = []
        view = self._view()
        if view.kind not in ("binary", "ternary"):
            return
        if view.kind == "binary":
            cong = [it for it in self._points() if it["kind"] == "congruent"]
            if cong:
                X, Y = to_plot(view, np.array([next(iter(c["cur_x"].values())) for c in cong]),
                               np.array([c["T"] for c in cong]))
                ax.scatter(X, Y, marker="D", s=30, facecolor="white", edgecolor="black", linewidths=1.0, zorder=7)
        if not self.visible:
            return
        for i, e in enumerate(self.entries):
            if (e["kind"] == "tietriangle" or (e["kind"] == "boundary" and e.get("direction") is not None)) \
                    != (view.kind == "ternary"):
                continue
            if list(e["elements"]) != list(view.elements):
                continue
            if e["kind"] == "tietriangle" and abs(e["T"] - view.temperature) > 1e-6:
                continue
            X, Y, keys, movable = self._xy(e)
            a = {"entry": i, "keys": keys, "movable": movable}
            if e["kind"] == "invariant":
                a["line"], = ax.plot([X.min(), X.max()], [Y[0], Y[0]], ls="--", lw=2.2, color=COLOR, zorder=8)
            elif e["kind"] == "tietriangle":
                a["line"], = ax.plot(np.r_[X, X[:1]], np.r_[Y, Y[:1]], ls="--", lw=2.0, color=COLOR, zorder=8)
            face = ["white" if m else "#c8c8c8" for m in movable]
            size = 150 if e["kind"] == "congruent" else 90
            a["handles"] = ax.scatter(X, Y, s=size, facecolor="none" if e["kind"] == "congruent" else face,
                                      edgecolor=COLOR, linewidths=2.2, zorder=9,
                                      marker="s" if e["kind"] == "boundary" else "o")
            if e["kind"] == "congruent":
                a["dot"] = ax.scatter(X, Y, s=18, color=COLOR, zorder=9)
            a["pred"] = ax.scatter([], [], s=28, color=COLOR, zorder=10, marker="o")
            a["label"] = ax.annotate(f"{i + 1}", (float(np.max(X)), float(Y[int(np.argmax(X))])), xytext=(8, 6),
                                     textcoords="offset points", fontsize=9, fontweight="bold", color="#a0006d",
                                     zorder=9)
            self._artists.append(a)

    def update_overlay(self):
        for e in self.entries:
            self._constrain(e)
        drawn = {a["entry"] for a in self._artists}
        want = {i for i, e in enumerate(self.entries)}
        if not self.visible or not drawn or drawn - want:
            self.win.replot()
            self.panel.refresh_values()
            return
        for a in self._artists:
            e = self.entries[a["entry"]]
            X, Y, _, _ = self._xy(e)
            if e["kind"] == "invariant":
                a["line"].set_data([X.min(), X.max()], [Y[0], Y[0]])
            elif e["kind"] == "tietriangle":
                a["line"].set_data(np.r_[X, X[:1]], np.r_[Y, Y[:1]])
            a["handles"].set_offsets(np.c_[X, Y])
            if "dot" in a:
                a["dot"].set_offsets(np.c_[X, Y])
            a["label"].xy = (float(np.max(X)), float(Y[int(np.argmax(X))]))
            if "pred" in a:
                P = self._predicted_xy(a["entry"])
                a["pred"].set_offsets(P if P is not None else np.empty((0, 2)))
        self.panel.refresh_values()
        self.win.canvas.draw_idle()

    def _predicted_xy(self, i: int):
        """Plot positions where the live linear model predicts entry i's feature to be."""
        if not self._pred:
            return None
        e = self.entries[i]
        view = self._view()
        T = self._pred.get((i, "T"), e["T"])
        get = lambda k: self._pred.get((i, k), e["cur_x"].get(k))
        if e["kind"] == "tietriangle":
            B, C = e["elements"][1], e["elements"][2]
            X, Y = to_plot(view, np.array([get(f"{n}|{B}") for n in e["names"]]),
                           np.array([get(f"{n}|{C}") for n in e["names"]]))
        elif e["kind"] == "congruent":
            X, Y = to_plot(view, np.array([get(e["names"][0])]), np.array([T]))
        elif e["kind"] == "boundary" and e.get("direction") is not None:
            els, a = e["elements"], e["names"][0]
            d = np.asarray(e["direction"])
            cur = np.array([e["cur_x"][f"{a}|{els[1]}"], e["cur_x"][f"{a}|{els[2]}"]])
            pred = self._pred.get((i, f"{a}|across"))
            if pred is None:
                return None
            q = cur + (pred - float(cur @ d)) * d
            X, Y = to_plot(view, np.array([q[0]]), np.array([q[1]]))
        elif e["kind"] == "boundary":
            X, Y = to_plot(view, np.array([get(e["names"][0])]), np.array([e["want_T"]]))
        else:
            X, Y = to_plot(view, np.array([get(n) for n in e["names"]]), np.full(len(e["names"]), T))
        return np.c_[X, Y]

    # ------------------------------------------------------------------------------------------ mouse
    def _ax(self):
        return self.win.figure.axes[0] if self.win.figure.axes else None

    def _on_press(self, ev):
        ax = self._ax()
        if ax is None or ev.inaxes is not ax or self.win.toolbar.mode:
            return
        if ev.button == 3:
            self._context_menu(ev, ax)
            return
        if ev.button != 1 or not self._artists:
            return
        best = None
        for a in self._artists:
            pts = ax.transData.transform(a["handles"].get_offsets())
            d = np.hypot(pts[:, 0] - ev.x, pts[:, 1] - ev.y)
            k = int(np.argmin(d))
            grab = GRAB_PX + (4 if a["keys"] == ["point"] else 0)
            if d[k] <= grab and (best is None or d[k] < best[0]):
                e = self.entries[a["entry"]]
                if not a["movable"][k]:
                    # a fixed-composition handle: on an invariant it moves the temperature line, else nothing
                    what = "line" if e["kind"] == "invariant" else None
                else:
                    what = a["keys"][k]
                if what is not None:
                    best = (d[k], (a["entry"], what))
        if best is None:
            for a in self._artists:
                e = self.entries[a["entry"]]
                if e["kind"] != "invariant":
                    continue
                (x0, x1), (y0, _) = a["line"].get_data()
                p0 = ax.transData.transform((x0, y0))
                p1 = ax.transData.transform((x1, y0))
                d = abs(ev.y - p0[1])
                if d <= GRAB_PX and min(p0[0], p1[0]) - GRAB_PX <= ev.x <= max(p0[0], p1[0]) + GRAB_PX:
                    if best is None or d < best[0]:
                        best = (d, (a["entry"], "line"))
        if best is not None:
            self._drag = best[1]
            self._live_start()

    def _binary_x(self, xd: float) -> float:
        view = self._view()
        A, B = view.elements
        return float(np.clip(U.binary_x_from_display(xd, A, B, view.masses, view.x_unit), 0, 1))

    def _ternary_x(self, X: float, Y: float):
        """Plot coordinates of a ternary section -> mole fractions (xB, xC)."""
        view = self._view()
        A, B, C = view.elements
        b, c = U.xy_to_tri(X, Y)
        b, c = float(np.clip(b, 0, 1)), float(np.clip(c, 0, 1))
        if U.is_mass(view.x_unit):
            fr = U.x_from_display({B: b, C: c}, [A, B, C], view.masses, "mass fraction")
            b, c = float(fr[B]), float(fr[C])
        return b, c

    def _on_motion(self, ev):
        if self._drag is None or ev.xdata is None or ev.ydata is None:
            return
        i, what = self._drag
        if i >= len(self.entries):
            return
        e = self.entries[i]
        view = self._view()
        if e["kind"] == "boundary" and e.get("direction") is not None:
            b, c = self._ternary_x(ev.xdata, ev.ydata)
            els = e["elements"]
            e["want_x"][f"node|{els[1]}"], e["want_x"][f"node|{els[2]}"] = b, c
        elif e["kind"] == "boundary":
            e["want_T"] = float(U.t_to_K(ev.ydata, view.t_unit))
            e["want_x"][e["names"][0]] = self._binary_x(ev.xdata)
        elif e["kind"] == "tietriangle":
            b, c = self._ternary_x(ev.xdata, ev.ydata)
            B, C = e["elements"][1], e["elements"][2]
            e["want_x"][f"{what}|{B}"], e["want_x"][f"{what}|{C}"] = b, c
        else:
            if what in ("line", "point"):
                e["want_T"] = float(U.t_to_K(ev.ydata, view.t_unit))
            if what == "point" and e["want_x"]:
                e["want_x"][e["names"][0]] = self._binary_x(ev.xdata)
            if what not in ("line", "point") and what in e["want_x"]:
                e["want_x"][what] = self._binary_x(ev.xdata)
        if self._live is not None and self._lin is not None:
            self._live_update()
        self.update_overlay()

    def _on_release(self, ev):
        was = self._drag
        self._drag = None
        if was is not None and self._live is not None:
            self._live_finish()

    def _pick(self, ev, ax):
        view = self._view()
        best = None
        for it in self._points():
            if it["kind"] == "invariant":
                X, Y = to_plot(view, np.array(list(it["cur_x"].values())), np.full(len(it["cur_x"]), it["T"]))
                p0 = ax.transData.transform((X.min(), Y[0]))
                p1 = ax.transData.transform((X.max(), Y[0]))
                d = np.hypot(max(p0[0] - ev.x, 0, ev.x - p1[0]), ev.y - p0[1])
            elif it["kind"] == "congruent":
                X, Y = to_plot(view, np.array([next(iter(it["cur_x"].values()))]), np.array([it["T"]]))
                p = ax.transData.transform((X[0], Y[0]))
                d = np.hypot(p[0] - ev.x, p[1] - ev.y) - 6
            else:
                B, C = it["elements"][1], it["elements"][2]
                X, Y = to_plot(view, np.array([it["cur_x"][f"{n}|{B}"] for n in it["names"]]),
                               np.array([it["cur_x"][f"{n}|{C}"] for n in it["names"]]))
                P = ax.transData.transform(np.c_[X, Y])
                d = _distance_to_triangle(P, np.array([ev.x, ev.y]))
            if d <= PICK_PX and (best is None or d < best[0]):
                best = (d, it)
        return None if best is None else best[1]

    def _pick_boundary(self, ev, ax) -> Optional[Dict]:
        """The point of a two-phase boundary nearest to a click (a boundary node)."""
        view = self._view()
        best = None
        for b in self._boundary_curves():
            X, Y = to_plot(view, b.x, b.y)
            P = ax.transData.transform(np.c_[X, Y])
            d = np.hypot(P[:, 0] - ev.x, P[:, 1] - ev.y)
            k = int(np.nanargmin(d))
            if d[k] <= PICK_PX and (best is None or d[k] < best[0]):
                best = (d[k], b, k)
        if best is None:
            return None
        _, b, k = best
        if abs(b.x[k] - b.partner_x[k]) < 1e-4:
            return None                                   # a congruent or pure-element end: not a boundary
        spaces = self.win._result_for_display(view).phase_space or {}
        if spaces.get(base_phase(b.phase), {}).get("rank", 1) == 0:
            return None                                   # a line compound's boundary cannot move sideways
        if view.kind == "ternary":
            if b.partner_y is None:
                return None
            els = list(view.elements)
            pa, pb = np.array([b.x[k], b.y[k]]), np.array([b.partner_x[k], b.partner_y[k]])
            if np.linalg.norm(pb - pa) < 1e-4:
                return None
            d = (pb - pa) / np.linalg.norm(pb - pa)
            return dict(kind="boundary", names=[b.phase, b.partner], T=float(view.temperature), spaces=spaces,
                        elements=els, direction=d.tolist(),
                        cur_x={f"{b.phase}|{els[1]}": float(pa[0]), f"{b.phase}|{els[2]}": float(pa[1]),
                               f"{b.partner}|{els[1]}": float(pb[0]), f"{b.partner}|{els[2]}": float(pb[1])})
        return dict(kind="boundary", names=[b.phase, b.partner], T=float(b.y[k]),
                    cur_x={b.phase: float(b.x[k]), b.partner: float(b.partner_x[k])},
                    spaces=spaces, elements=list(view.elements))

    def _context_menu(self, ev, ax):
        view = self._view()
        menu = QMenu(self.win)
        if view.kind in ("binary", "ternary"):
            it = self._pick(ev, ax)
            node = self._pick_boundary(ev, ax) if it is None else None
            if node is not None:
                where = (f"at {U.t_from_K(node['T'], view.t_unit):.6g} {view.t_unit}" if view.kind == "binary"
                         else "of this section")
                menu.addAction(f"Add boundary node: {node['names'][0]} boundary of {node['names'][0]} + "
                               f"{node['names'][1]} {where}").triggered.connect(lambda: self.add(node))
            if it is not None:
                idx = self._index_of(it)
                if idx is None:
                    if it["kind"] == "tietriangle":
                        text = f"Optimize tie triangle {' + '.join(it['names'])}…"
                    else:
                        T = f"{U.t_from_K(it['T'], view.t_unit):.6g} {view.t_unit}"
                        what = "invariant" if it["kind"] == "invariant" else "congruent point"
                        label = entry_label(it).replace("  (congruent)", "")
                        text = f"Optimize {what} {label} ({T})…"
                    fx = fixed_phases(it)
                    if it["kind"] == "congruent" and it.get("fixed_x"):
                        fx = [it["compound"]]
                    if fx:
                        text = text[:-1] + f" — fixed composition: {', '.join(fx)}…"
                    menu.addAction(text).triggered.connect(lambda: self.add(it))
                else:
                    menu.addAction(f"Remove target {idx + 1} ({entry_label(it)})").triggered.connect(
                        lambda: self.remove(idx))
        self._mark_menu(menu, ev, ax)
        probe = self._probe_point(ev)
        if probe is not None:
            menu.addSeparator()
            menu.addAction("Equilibrium here").triggered.connect(lambda: self.win.probe(*probe))
            T, comp = probe
            if view.kind == "binary":
                menu.addAction(f"G–x curve at {U.t_from_K(T, view.t_unit):.6g} {view.t_unit}").triggered.connect(
                    lambda: self.win.open_gx(T))
            menu.addAction("Step plot at this composition").triggered.connect(lambda: self.win.open_step(comp))
        menu.addSeparator()
        menu.addAction("Export figure…").triggered.connect(self.win.export_figure)
        if self.entries:
            menu.addSeparator()
            if self.visible:
                menu.addAction("Hide optimization targets").triggered.connect(self._hide)
            else:
                menu.addAction("Show optimization targets").triggered.connect(self._show)
            menu.addAction("Clear all targets").triggered.connect(self.clear)
        menu.exec(QCursor.pos())

    def _probe_point(self, ev):
        """(T in K, {element: mole fraction}) of a click, or None outside the diagram."""
        view = self._view()
        if ev.xdata is None or ev.ydata is None:
            return None
        els = list(view.elements)
        if view.kind == "binary":
            return float(U.t_to_K(ev.ydata, view.t_unit)), {els[1]: self._binary_x(ev.xdata)}
        if view.kind == "ternary":
            b, c = self._ternary_x(ev.xdata, ev.ydata)
            if b + c > 1 or b <= 0 or c <= 0:
                return None
            return view.temperature, {els[1]: b, els[2]: c}
        if view.kind == "isopleth":
            A, B, C = els
            if U.is_mass(view.x_unit):
                return None              # an isopleth's wt.% axis depends on all three elements: not converted here
            xb = float(ev.xdata) / (100.0 if view.x_unit.endswith("%") else 1.0)
            return float(U.t_to_K(ev.ydata, view.t_unit)), {B: xb, C: view.fixed}
        return None

    def export_targets(self):
        from PyQt6.QtWidgets import QFileDialog
        from .espei_export import dataset_dicts, write
        if not self.entries:
            self.panel.status.setText("No targets to export.")
            return
        folder = QFileDialog.getExistingDirectory(self.win, "Folder for the ESPEI datasets")
        if not folder:
            return
        import os
        paths = []
        for els, d in dataset_dicts(self.entries).items():
            paths.append(write(os.path.join(folder, f"tdbtweak-targets-{'-'.join(els)}.json"), d))
        self.win._log("exported targets as ESPEI ZPF data: " + ", ".join(paths))
        self.panel.status.setText(f"Wrote {len(paths)} dataset(s) to {folder}.")

    def _mark_menu(self, menu, ev, ax):
        """Right-click: mark a data point here, interpreted from the nearest computed boundary."""
        view = self._view()
        if view.kind not in ("binary", "ternary") or ev.xdata is None:
            return
        best = None
        for b in self._boundary_curves():
            X, Y = to_plot(view, b.x, b.y)
            P = ax.transData.transform(np.c_[X, Y])
            d = np.hypot(P[:, 0] - ev.x, P[:, 1] - ev.y)
            k = int(np.nanargmin(d))
            if best is None or d[k] < best[0]:
                best = (d[k], b)
        pt = self._probe_point(ev)
        if best is None or pt is None:
            return
        b = best[1]
        T, comp = pt
        sub = menu.addMenu("Mark data point here")
        for ph, partner in ((b.phase, b.partner), (b.partner, b.phase)):
            sub.addAction(f"{ph} boundary of {ph} + {partner}").triggered.connect(
                lambda _=False, ph=ph, partner=partner: self.win.mark_point(T, comp, ph, partner))

    def _hide(self):
        self.visible = False
        self.win.replot()

    def _show(self):
        self.visible = True
        self.win.replot()

    # ------------------------------------------------------------------------------------------ running
    def _job_inputs(self):
        """(request, variables, targets, settings) of an optimisation of the current targets, or an error text."""
        w = self.win
        if w.doc is None or not self.entries:
            return "no targets"
        req = next((e["req"] for e in self.entries if e.get("req") is not None), None) or w.build_request()
        if not self.panel._rows:
            self.fill_variables(selection=False)
        chosen = self.panel.chosen()
        if not chosen:
            return "Choose at least one parameter to vary."
        for r, _ in chosen:
            if r.kind == "COEF" and r.rid not in w.doc.live:
                return "Map once after choosing coefficient parameters (they are compiled first)."
        variables = [self._variable(r, lim) for r, lim in chosen]
        targets = self.panel.build_targets()
        settings = self.panel.settings()
        settings.zpf_data = [d for d in w.zpf if d.enabled] if self.panel.fit_zpf.isChecked() else []
        return req, variables, targets, settings

    def _variable(self, r, lim):
        doc = self.win.doc
        phases = {x.phase for x in doc.rows if x.symbol == r.symbol and x.phase}
        atoms = max((doc.phase_atoms(p) for p in phases), default=1.0)
        return Variable(r.symbol, r.basis, f"{r.symbol if r.kind == 'VV' else ''} {r.meaning()}".strip(), lim,
                        atoms=atoms)

    # live drag ------------------------------------------------------------------------------------------
    def _live_key(self):
        from .views import values_stamp
        w = self.win
        return (values_stamp(w.doc.compute_text(), w.doc.parameter_values()),
                tuple((e["kind"], tuple(e["names"]), round(e["T"], 3)) for e in self.entries),
                tuple(r.symbol for r, _ in self.panel.chosen()), self.panel.hold_others.isChecked(),
                self.panel.fit_zpf.isChecked(), self.panel.norm_weight())

    def _live_start(self):
        """At a press: make sure a linear model for the current parameters exists (or is being computed)."""
        w = self.win
        if not self.panel.live.isChecked() or w.runner.running:
            return
        key = self._live_key()
        self._live = {"start": {r.symbol: w.doc.value(r) for r, _ in self.panel.chosen()}, "applied": False}
        if self._lin is not None and self._lin_key == key:
            return
        inp = self._job_inputs()
        if isinstance(inp, str):
            self._live = None
            return
        req, variables, targets, settings = inp
        # the user's targets at their current positions: the model is linearised at the current state
        for t, e in zip(targets, self.entries):
            if e["kind"] != "tietriangle" and e["kind"] != "boundary":
                t.T_target = e["T"]
            if e["kind"] == "boundary" and e.get("direction") is not None:
                els, a = e["elements"], e["names"][0]
                cur = np.array([e["cur_x"][f"{a}|{els[1]}"], e["cur_x"][f"{a}|{els[2]}"]])
                t.point = tuple(cur)
                t.x_target = {f"{a}|across": float(cur @ np.asarray(e["direction"]))}
                continue
            t.x_target = {k: (e["cur_x"].get(k, v) if v is not None else None) for k, v in t.x_target.items()}
        settings.jacobian_only = True
        settings.reject_artifacts = False
        self._lin, self._lin_key = None, key
        self._lin_wait = key
        self.panel.status.setText("live drag: computing how the targets respond to the parameters…")
        w.runner.submit_opt(w.doc.compute_text(), list(w.doc.parameter_values()), w.doc.parameter_values(), req,
                            (variables, targets, settings))

    def _on_linear(self, res):
        if getattr(self, "_lin_wait", None) is None:
            return
        self._lin = res.linear
        self._lin["max_total"] = 1.0              # trust region: one per-iteration limit of change
        self._lin_wait = None
        self.panel.status.setText(f"live drag ready ({len(res.linear['names'])} parameters, "
                                  f"{len(res.holds)} held features): drag the handles")
        self.win.cancel_btn.setEnabled(False)
        if self._drag is not None:
            self._live_update()

    def _live_update(self):
        """Apply the linear model for the current handle positions (called on every mouse move)."""
        w = self.win
        if self._lin is None or self._live is None:
            return
        wanted = {}
        for i, e in enumerate(self.entries):
            if e["kind"] not in ("tietriangle", "boundary"):
                wanted[(i, "T")] = e["want_T"]
            if e["kind"] == "boundary" and e.get("direction") is not None:
                els = e["elements"]
                p = np.array([e["want_x"][f"node|{els[1]}"], e["want_x"][f"node|{els[2]}"]])
                wanted[(i, f"{e['names'][0]}|across")] = float(p @ np.asarray(e["direction"]))
                continue
            for k, val in e["want_x"].items():
                wanted[(i, k)] = val
        vals, pred = linear_step(self._lin, wanted)
        rows = {r.symbol: r for r in w.doc.rows if r.symbol in vals}
        for name, v in vals.items():
            r = rows.get(name)
            if r is not None and not w.doc.is_locked(r):
                w.doc.set_value(r, v)
        self._live["applied"] = True
        self._pred = {k: v for k, v in pred.items() if k[0] < len(self.entries)}
        w.params.model.refresh_values()
        w._nudge_timer.start()                   # fast redraw when the mouse pauses

    def _live_finish(self):
        """At release: one undo step from the values before the drag to the values now."""
        w = self.win
        live, self._live = self._live, None
        self._pred = {}
        if live is None or not live["applied"]:
            return
        final = {s_: w.doc.value(next(r for r in w.doc.rows if r.symbol == s_)) for s_ in live["start"]}
        rows = {r.symbol: r for r in w.doc.rows if r.symbol in live["start"]}
        for s_, v in live["start"].items():
            w.doc.set_value(rows[s_], v)
        apply_symbol_values(w.params, final, "live drag")
        self._lin, self._lin_key = None, None     # the parameters changed: the next press linearises again
        w._nudge_timer.stop()
        w.redraw(fast=True)

    def run(self):
        w = self.win
        fit = self.panel.fit_zpf.isChecked()
        if w.doc is None or (not self.entries and not fit):
            self.panel.status.setText("Add a target first (right-click an invariant, congruent point or tie triangle), "
                                      "or tick “Fit the loaded ZPF data”.")
            return
        if fit and not any(d.enabled for d in w.zpf):
            self.panel.status.setText("Load ZPF data first (ZPF data tab).")
            return
        req = next((e["req"] for e in self.entries if e.get("req") is not None), None) or w.build_request()
        if req is None:
            self.panel.status.setText("No diagram to optimise on.")
            return
        bad = [entry_label(e) for e in self.entries
               if any(el not in w.subsystem for el in e.get("elements", []))]
        if bad:
            self.panel.status.setText("Targets with elements outside the edited set: " + "; ".join(bad))
            return
        if not self.panel._rows:
            self.fill_variables(selection=False)
        chosen = self.panel.chosen()
        if not chosen:
            self.panel.status.setText("Choose at least one parameter to vary.")
            return
        targets = self.panel.build_targets()
        empty = [target_label(t) for t in targets
                 if t.T_target is None and not any(v is not None for v in t.x_target.values())]
        if empty:
            self.panel.status.setText("Tick at least one value of: " + ", ".join(empty))
            return
        rebuild = False
        for r, _ in chosen:
            if r.kind == "COEF" and r.rid not in w.doc.live:
                w.doc.live.add(r.rid)
                rebuild = True
        variables = [self._variable(r, lim) for r, lim in chosen]
        settings = self.panel.settings()
        settings.zpf_data = [d for d in w.zpf if d.enabled] if fit else []
        self.panel.clear_history()
        self.panel.set_running(True, "optimizing…" + (" (compiling the chosen coefficients first)" if rebuild else ""))
        w.runner.submit_opt(w.doc.compute_text(), list(w.doc.parameter_values()), w.doc.parameter_values(), req,
                            (variables, targets, settings))
        w.cancel_btn.setEnabled(True)

    def _on_progress(self, it):
        self.panel.add_iteration(it)
        self.panel.status.setText(f"iteration {it.index}: max |residual| {it.residual:.3g} tolerances"
                                  + (f" — {it.note}" if it.note else ""))

    def _on_finished(self, res):
        w = self.win
        if getattr(res, "linear", None) is not None:
            self._on_linear(res)
            return
        w.cancel_btn.setEnabled(False)
        k = len(res.iterations) - 1
        self.panel.set_result(res, k)
        warn = [f for f in getattr(res, "artifacts", []) if f.severity == "warning"]
        extra = ""
        if warn:
            extra = "<br><span style='color:#b35c00'>⚠ " + "<br>⚠ ".join(f.text for f in warn[:4]) + "</span>"
            for f in res.artifacts:
                w._log(f"artifact check ({f.severity}): {f.text}")
        self.panel.set_running(False, f"{res.message} ({res.seconds:.1f} s). Applied iteration {k}; select "
                                      f"another and press Apply, or Ctrl+Z to undo.{extra}")
        w._log(f"optimize {len(res.targets)} target(s) ({'; '.join(target_label(t) for t in res.targets)}): "
               f"{res.message}, {len(res.variables)} parameters")
        self.apply_iteration(k, res)

    def failed(self, msg: str):
        self.panel.set_running(False, "<span style='color:#b00'>" + msg.split("\n")[0] + "</span>")

    def cancelled(self):
        self.panel.set_running(False, "stopped")

    def apply_iteration(self, k: int, res=None):
        res = res or self.panel.result
        if res is None or not (0 <= k < len(res.iterations)):
            return
        w = self.win
        n = apply_symbol_values(w.params, res.iterations[k].values,
                                f"optimize {len(res.targets)} target(s): iteration {k}")
        self.panel.update_variable_values(w.doc.parameter_values())
        w._log(f"applied optimizer iteration {k} ({n} values changed)")
        w.redraw()


def _distance_to_triangle(P: np.ndarray, q: np.ndarray) -> float:
    """0 inside the triangle P (3x2, pixels), else the distance to its nearest edge."""
    a, b, c = P
    v0, v1, v2 = c - a, b - a, q - a
    den = v0[0] * v1[1] - v1[0] * v0[1]
    if abs(den) > 1e-12:
        u = (v2[0] * v1[1] - v1[0] * v2[1]) / den
        w = (v0[0] * v2[1] - v2[0] * v0[1]) / den
        if u >= 0 and w >= 0 and u + w <= 1:
            return 0.0
    best = np.inf
    for p0, p1 in ((a, b), (b, c), (c, a)):
        seg = p1 - p0
        t = float(np.clip(np.dot(q - p0, seg) / max(np.dot(seg, seg), 1e-12), 0, 1))
        best = min(best, float(np.linalg.norm(q - (p0 + t * seg))))
    return best
