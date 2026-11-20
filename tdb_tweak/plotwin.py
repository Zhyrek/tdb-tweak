"""Small companion plot windows (G-x curves, step plots) that follow parameter edits, and figure export."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCursor
from PyQt6.QtWidgets import QFileDialog, QLabel, QMenu, QVBoxLayout, QWidget

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from . import units as U
from .diagram import PhaseColors


def export_figure(parent, fig, default_name: str = "figure.png"):
    path, _ = QFileDialog.getSaveFileName(parent, "Export figure", default_name,
                                          "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
    if path:
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


class PlotWindow(QWidget):
    """A tool window with a matplotlib figure; right-click: Export figure…"""
    kind = ""

    def __init__(self, title: str, colors: PhaseColors, parent=None):
        super().__init__(parent, Qt.WindowType.Tool)
        self.setWindowTitle(title)
        self.resize(720, 520)
        self.colors = colors
        self.figure = Figure(figsize=(7, 5))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.status = QLabel("computing…")
        self.status.setStyleSheet("color:#666;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas, 1)
        lay.addWidget(self.status)
        self.canvas.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self._menu)
        self.t_unit = "K"
        self.x_unit_pct = True

    def _menu(self, pos):
        m = QMenu(self)
        m.addAction("Export figure…").triggered.connect(
            lambda: export_figure(self, self.figure, self.windowTitle().replace(" ", "_") + ".png"))
        m.exec(QCursor.pos())

    def task(self):
        """(kind, args) of the computation to run with the current parameters."""
        raise NotImplementedError

    def show_result(self, res):
        raise NotImplementedError

    def show_error(self, msg: str):
        self.status.setText(f"<span style='color:#b00'>{msg}</span>")


class GxWindow(PlotWindow):
    kind = "gx"

    def __init__(self, req, T: float, colors: PhaseColors, t_unit: str, parent=None):
        A, B = req.elements[0], req.elements[1]
        super().__init__(f"G–x {A.capitalize()}-{B.capitalize()} at {U.t_from_K(T, t_unit):.6g} {t_unit}",
                         colors, parent)
        self.req, self.T, self.t_unit = req, float(T), t_unit
        self._limits = None

    def task(self):
        return "gx", {"T": self.T}

    def show_result(self, r: Dict):
        ax0 = self.figure.axes[0] if self.figure.axes else None
        keep = (tuple(ax0.get_xlim()), tuple(ax0.get_ylim())) if (ax0 is not None and self._limits is not None and
                                                                  (tuple(ax0.get_xlim()), tuple(ax0.get_ylim())) != self._limits) else None
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        gA, gB = r["ref"]
        ref = lambda x: (1 - x) * gA + x * gB
        A, B = r["elements"]
        lo, hi = [], []
        for ph, (x, g) in sorted(r["curves"].items()):
            d = (g - ref(x)) / 1000.0
            hi.append(np.nanmax(d))
            if len(x) == 1:
                ax.scatter(x * 100, d, s=40, color=self.colors(ph), zorder=4, label=ph, edgecolor="black", lw=0.6)
            else:
                ax.plot(x * 100, d, color=self.colors(ph), lw=1.6, label=ph)
            lo.append(np.nanmin(d))
        for x1, g1, x2, g2, phs in r["tangents"]:
            ax.plot([x1 * 100, x2 * 100], [(g1 - ref(x1)) / 1000, (g2 - ref(x2)) / 1000], color="black", lw=1.0,
                    ls="--", zorder=5)
            ax.scatter([x1 * 100, x2 * 100], [(g1 - ref(x1)) / 1000, (g2 - ref(x2)) / 1000], color="black", s=12,
                       zorder=6)
        ax.set_xlim(0, 100)
        if lo:
            bottom, top = min(lo), max(hi)
            pad = 0.12 * max(abs(bottom), 1.0)
            ax.set_ylim(bottom - pad, min(top, 0.0) + pad)
        ax.axhline(0, color="#bbb", lw=0.6)
        ax.set_xlabel(f"{B.capitalize()} (at.%)")
        ax.set_ylabel("G − G(pure-element line)  (kJ/mol)")
        ax.set_title(f"{U.t_from_K(r['T'], self.t_unit):.6g} {self.t_unit}   (dashed: common tangents of the equilibrium)",
                     fontsize=9)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        if keep:
            ax.set_xlim(*keep[0])
            ax.set_ylim(*keep[1])
        self._limits = (tuple(ax.get_xlim()), tuple(ax.get_ylim()))
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.status.setText("Follows your parameter edits.")


class StepWindow(PlotWindow):
    kind = "step"

    def __init__(self, req, comp: Dict[str, float], tmin: float, tmax: float, colors: PhaseColors, t_unit: str,
                 parent=None):
        txt = ", ".join(f"{e.capitalize()} {x * 100:.3g}" for e, x in comp.items())
        super().__init__(f"Step plot at {txt} at.%", colors, parent)
        self.req, self.comp, self.tmin, self.tmax, self.t_unit = req, dict(comp), float(tmin), float(tmax), t_unit

    def task(self):
        return "step", {"comp": self.comp, "tmin": self.tmin, "tmax": self.tmax}

    def show_result(self, r: Dict):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        T = U.t_from_K(r["T"], self.t_unit)
        for ph, f in sorted(r["fractions"].items()):
            ax.plot(T, f, color=self.colors(ph), lw=1.6, label=ph)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlim(float(T[0]), float(T[-1]))
        ax.set_xlabel(f"Temperature ({self.t_unit})")
        ax.set_ylabel("Phase fraction")
        txt = ", ".join(f"x({e.capitalize()}) = {x * 100:.3g} %" for e, x in r["composition"].items())
        ax.set_title(txt, fontsize=9)
        ax.legend(fontsize=8, frameon=False)
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self.status.setText("Follows your parameter edits.")
