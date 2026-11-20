"""tdb-tweak main window."""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QSettings, Qt, QTimer
from PyQt6.QtGui import QAction, QColor, QFont, QKeySequence, QShortcut
from PyQt6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDockWidget,
                             QDoubleSpinBox, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QListWidget,
                             QListWidgetItem, QMainWindow, QMenu, QMessageBox, QPlainTextEdit, QPushButton,
                             QSizePolicy, QSlider, QSpinBox, QTabWidget, QToolBar, QToolButton, QVBoxLayout,
                             QWidget)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from . import __version__
from . import units as U
from .calibdialog import CalibrationDialog, load_rgba
from .calibration import Calibration
from .diagram import PhaseColors, View, draw_all
from .engine import MapRequest, MapResult, cg_available
from .paramtable import ParamPanel
from .tdbmodel import TDBDocument
from .views import ResultCache, SavedView, auto_name, request_key, same_diagram, values_stamp
from .worker import MapRunner
from .zpf import ZPFDataset, load_paths, matching, temperatures

KINDS = [("binary", "Binary T–x"), ("ternary", "Ternary isothermal"), ("isopleth", "Isopleth")]


def _spin(lo, hi, val, step, decimals=2, suffix=""):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(val)
    s.setKeyboardTracking(False)
    if suffix:
        s.setSuffix(suffix)
    s.setMinimumWidth(80)
    return s


class CheckListDialog(QDialog):
    def __init__(self, title: str, items: List[str], checked: set, parent=None, min_checked: int = 1):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.min_checked = min_checked
        self.list = QListWidget()
        for it in items:
            w = QListWidgetItem(it)
            w.setFlags(w.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            w.setCheckState(Qt.CheckState.Checked if it in checked else Qt.CheckState.Unchecked)
            self.list.addItem(w)
        allb, noneb = QPushButton("All"), QPushButton("None")
        allb.clicked.connect(lambda: self._set_all(True))
        noneb.clicked.connect(lambda: self._set_all(False))
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._ok)
        bb.rejected.connect(self.reject)
        row = QHBoxLayout()
        row.addWidget(allb)
        row.addWidget(noneb)
        row.addStretch(1)
        lay = QVBoxLayout(self)
        lay.addWidget(self.list)
        lay.addLayout(row)
        lay.addWidget(bb)
        self.resize(320, 480)

    def _set_all(self, on: bool):
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)

    def checked(self) -> List[str]:
        return [self.list.item(i).text() for i in range(self.list.count())
                if self.list.item(i).checkState() == Qt.CheckState.Checked]

    def _ok(self):
        if len(self.checked()) < self.min_checked:
            QMessageBox.warning(self, self.windowTitle(), f"Choose at least {self.min_checked}.")
            return
        self.accept()


class ImageEntry:
    def __init__(self, cal: Calibration):
        self.cal = cal
        self._rgba: Optional[np.ndarray] = None
        self._rect = None
        self._rect_key = None

    @property
    def rgba(self) -> np.ndarray:
        if self._rgba is None:
            self._rgba = load_rgba(self.cal.image_path)
        return self._rgba

    def rectified(self):
        key = json.dumps(self.cal.to_dict(), sort_keys=True, default=str)
        if key != self._rect_key:
            self._rect = self.cal.rectify(self.rgba)
            self._rect_key = key
        return self._rect


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("tdb-tweak")
        self.resize(1600, 950)
        self.settings = QSettings("tdb-tweak", "tdb-tweak")
        self.doc: Optional[TDBDocument] = None
        self.phase_db = None
        self.subsystem: List[str] = []
        self.excluded_phases: set = set()
        self.images: List[ImageEntry] = []
        self.zpf: List[ZPFDataset] = []
        self.result: Optional[MapResult] = None
        self.result_values_stamp = None
        self.colors = PhaseColors()
        self.stale = False
        self._view_key = None
        self._zoom = None              # user zoom kept across redraws of the same diagram
        self._last_drawn = None        # limits after the last draw (to detect zooming)
        self._pending_zoom = None      # limits to apply on the next draw (from a saved view)
        self.views: List[SavedView] = []
        self._active_view: Optional[int] = None
        self.cache = ResultCache()
        self._queue: List[Tuple[MapRequest, str]] = []   # (request, label) still to map ("Map all views")
        self._pending = None           # (request key, stamp, label) of the running job
        self._zpf_artists = []
        self._picked_T: Optional[float] = None
        self._map_started = 0.0

        self.runner = MapRunner(self)
        self.runner.progress.connect(self._on_progress)
        self.runner.finished.connect(self._on_finished)
        self.runner.failed.connect(self._on_failed)
        self.runner.cancelled.connect(self._on_cancelled)
        self.runner.status.connect(lambda s: self.statusBar().showMessage(s, 8000))

        self._build_ui()
        self._build_menus()
        self._update_kind_widgets()

    # ===================================================================================== UI construction
    def _build_ui(self):
        # --- parameter dock ---
        self.params = ParamPanel()
        self.params.changed.connect(self._on_values_changed)
        self.params.committed.connect(self._on_committed)
        self.params.subsystemRequested.connect(self.choose_subsystem)
        dock = QDockWidget("Parameters", self)
        dock.setObjectName("params")
        dock.setWidget(self.params)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self.resizeDocks([dock], [560], Qt.Orientation.Horizontal)

        # --- diagram controls ---
        self.kind = QComboBox()
        for k, label in KINDS:
            self.kind.addItem(label, k)
        self.el = [QComboBox() for _ in range(3)]
        self.el_label = [QLabel("A"), QLabel("x:"), QLabel("C")]
        self.t_unit = QComboBox()
        self.t_unit.addItems(U.T_UNITS)
        self.x_unit = QComboBox()
        self.x_unit.addItems(U.X_UNITS)
        self.solver = QComboBox()
        self.solver.addItem("pycalphad solver", False)
        self.solver.addItem("column generation (pycalphad-cg)", True)
        if not cg_available():
            self.solver.model().item(1).setEnabled(False)
            self.solver.setToolTip("pycalphad-cg is not installed: pip install pycalphad-cg")
        self.phases_btn = QPushButton("Phases…")
        self.redraw_btn = QPushButton("Redraw  (F5)")
        self.redraw_btn.setStyleSheet("font-weight: bold;")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.auto_cb = QCheckBox("Redraw on commit")
        self.auto_cb.setToolTip("Map again each time a value is committed (Enter) in the table")
        self.state_lbl = QLabel("")

        self.tmin = _spin(-500, 10000, 300, 50, 2)
        self.tmax = _spin(-500, 10000, 2000, 50, 2)
        self.tstep = _spin(0.1, 500, 10, 1, 2)
        self.tsec = _spin(-500, 10000, 1000, 10, 2)
        self.xmin = _spin(0, 100, 0, 1, 3, " %")
        self.xmax = _spin(0, 100, 100, 1, 3, " %")
        self.xstep = _spin(0.01, 20, 1, 0.5, 3, " %")
        self.ymin = _spin(0, 100, 0, 1, 3, " %")
        self.ymax = _spin(0, 100, 100, 1, 3, " %")
        self.fixed = _spin(0, 100, 10, 1, 3, " %")
        self.pdens = QSpinBox()
        self.pdens.setRange(20, 5000)
        self.pdens.setValue(500)
        self.pdens.setSingleStep(50)
        self.pdens.setToolTip("Sampling density of mapping's global-minimum check (pycalphad default 500).\n"
                              "It takes most of the mapping time: lower is faster but may miss phases.")
        self.tielines = QCheckBox("Tie-lines")
        self.fit_btn = QPushButton("Fit ranges to image")
        self.fit_btn.setToolTip("Map only the region shown by the visible literature image (faster)")
        self.reset_view_btn = QPushButton("Reset view")

        row1 = QHBoxLayout()
        row1.addWidget(self.kind)
        for lab, c in zip(self.el_label, self.el):
            row1.addWidget(lab)
            row1.addWidget(c)
        row1.addSpacing(10)
        row1.addWidget(QLabel("Units:"))
        row1.addWidget(self.t_unit)
        row1.addWidget(self.x_unit)
        row1.addSpacing(10)
        row1.addWidget(self.solver)
        row1.addWidget(self.phases_btn)
        row1.addStretch(1)
        row1.addWidget(self.state_lbl)
        row1.addWidget(self.auto_cb)
        row1.addWidget(self.cancel_btn)
        row1.addWidget(self.redraw_btn)

        self.row2 = QHBoxLayout()
        self.lbl_T = QLabel("T from")
        self.lbl_Tto = QLabel("to")
        self.lbl_Tstep = QLabel("step")
        self.lbl_Tsec = QLabel("T")
        self.lbl_x = QLabel("x from")
        self.lbl_xto = QLabel("to")
        self.lbl_xstep = QLabel("step")
        self.lbl_y = QLabel("C from")
        self.lbl_yto = QLabel("to")
        self.lbl_fixed = QLabel("fixed")
        for w in (self.lbl_T, self.tmin, self.lbl_Tto, self.tmax, self.lbl_Tstep, self.tstep, self.lbl_Tsec,
                  self.tsec, self.lbl_x, self.xmin, self.lbl_xto, self.xmax, self.lbl_y, self.ymin, self.lbl_yto,
                  self.ymax, self.lbl_xstep, self.xstep, self.lbl_fixed, self.fixed):
            self.row2.addWidget(w)
        self.row2.addSpacing(10)
        self.row2.addWidget(QLabel("Global-min pdens"))
        self.row2.addWidget(self.pdens)
        self.row2.addWidget(self.tielines)
        self.row2.addStretch(1)
        self.row2.addWidget(self.fit_btn)
        self.row2.addWidget(self.reset_view_btn)

        self.figure = Figure(figsize=(9, 6))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.canvas.mpl_connect("pick_event", self._on_pick)

        central = QWidget()
        cl = QVBoxLayout(central)
        cl.setContentsMargins(4, 4, 4, 4)
        cl.addLayout(row1)
        cl.addLayout(self.row2)
        cl.addWidget(self._build_views_bar())
        cl.addWidget(self.toolbar)
        cl.addWidget(self.canvas, 1)
        self.setCentralWidget(central)

        # --- overlay docks ---
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_images_tab(), "Literature images")
        self.tabs.addTab(self._build_zpf_tab(), "ZPF data")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.tabs.addTab(self.log, "Log")
        odock = QDockWidget("Overlays", self)
        odock.setObjectName("overlays")
        odock.setWidget(self.tabs)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, odock)
        self.resizeDocks([odock], [330], Qt.Orientation.Horizontal)

        # --- signals ---
        self.kind.currentIndexChanged.connect(self._on_kind_changed)
        for c in self.el:
            c.currentIndexChanged.connect(self._on_elements_changed)
        self.t_unit.currentTextChanged.connect(self._on_t_unit_changed)
        self.x_unit.currentTextChanged.connect(lambda *_: self.replot())
        self.tsec.valueChanged.connect(lambda *_: self._on_section_T_changed())
        self.tielines.toggled.connect(lambda *_: self.replot())
        self.phases_btn.clicked.connect(self.choose_phases)
        self.redraw_btn.clicked.connect(self.redraw)
        self.cancel_btn.clicked.connect(self.runner.cancel)
        self.fit_btn.clicked.connect(self.fit_ranges_to_image)
        self.reset_view_btn.clicked.connect(self.reset_view)
        for sp in (self.tmin, self.tmax, self.tstep, self.xmin, self.xmax, self.xstep, self.ymin, self.ymax,
                   self.fixed, self.pdens):
            sp.valueChanged.connect(lambda *_: self._update_state_label())
        self.solver.currentIndexChanged.connect(lambda *_: self._update_state_label())
        self._t_unit_prev = self.t_unit.currentText()
        self.statusBar().showMessage("Open a TDB file (File ▸ Open TDB…)")

    def _build_images_tab(self) -> QWidget:
        w = QWidget()
        self.img_list = QListWidget()
        self.img_list.itemChanged.connect(self._on_image_toggled)
        self.img_list.currentRowChanged.connect(self._on_image_selected)
        self.img_list.itemDoubleClicked.connect(lambda *_: self.calibrate_image())
        add, cal, rem = QPushButton("Add…"), QPushButton("Calibrate…"), QPushButton("Remove")
        add.clicked.connect(self.add_image)
        cal.clicked.connect(self.calibrate_image)
        rem.clicked.connect(self.remove_image)
        self.alpha = QSlider(Qt.Orientation.Horizontal)
        self.alpha.setRange(5, 100)
        self.alpha.setValue(60)
        self.alpha.valueChanged.connect(self._on_alpha)
        row = QHBoxLayout()
        row.addWidget(add)
        row.addWidget(cal)
        row.addWidget(rem)
        arow = QHBoxLayout()
        arow.addWidget(QLabel("Opacity"))
        arow.addWidget(self.alpha)
        note = QLabel("Checking an image switches the diagram to its type, elements and units. "
                      "Calibrations are also saved next to the image (*.tdbtweak.json).")
        note.setWordWrap(True)
        note.setStyleSheet("color:#666;")
        lay = QVBoxLayout(w)
        lay.addLayout(row)
        lay.addWidget(self.img_list, 1)
        lay.addLayout(arow)
        lay.addWidget(note)
        return w

    def _build_zpf_tab(self) -> QWidget:
        w = QWidget()
        self.zpf_list = QListWidget()
        self.zpf_list.itemChanged.connect(self._on_zpf_toggled)
        addf, addd, rem, clr = QPushButton("Files…"), QPushButton("Folder…"), QPushButton("Remove"), QPushButton("Clear")
        addf.clicked.connect(self.add_zpf_files)
        addd.clicked.connect(self.add_zpf_folder)
        rem.clicked.connect(self.remove_zpf)
        clr.clicked.connect(self.clear_zpf)
        self.zpf_tol = _spin(0, 500, 5, 1, 1, " K")
        self.zpf_tol.valueChanged.connect(lambda *_: (self._refresh_zpf_temps(), self.replot()))
        self.zpf_temps = QListWidget()
        self.zpf_temps.itemDoubleClicked.connect(self._on_zpf_temp_activated)
        self.map_picked_btn = QPushButton("Map section at picked point's T")
        self.map_picked_btn.setEnabled(False)
        self.map_picked_btn.clicked.connect(self._map_at_picked)
        row = QHBoxLayout()
        for b in (addf, addd, rem, clr):
            row.addWidget(b)
        trow = QHBoxLayout()
        trow.addWidget(QLabel("Ternary: show data within ±"))
        trow.addWidget(self.zpf_tol)
        lay = QVBoxLayout(w)
        lay.addLayout(row)
        lay.addWidget(self.zpf_list, 2)
        lay.addLayout(trow)
        lay.addWidget(QLabel("Temperatures in the data (double-click to map that section):"))
        lay.addWidget(self.zpf_temps, 1)
        lay.addWidget(self.map_picked_btn)
        return w

    def _build_menus(self):
        mb = self.menuBar()
        fm = mb.addMenu("&File")
        self._act(fm, "Open TDB…", self.open_tdb_dialog, QKeySequence.StandardKey.Open)
        self._act(fm, "Save TDB", self.save_tdb, QKeySequence.StandardKey.Save)
        self._act(fm, "Save TDB As…", self.save_tdb_as, QKeySequence("Ctrl+Shift+S"))
        fm.addSeparator()
        self._act(fm, "Open Project…", self.open_project_dialog, QKeySequence("Ctrl+Shift+O"))
        self._act(fm, "Save Project…", self.save_project_dialog, QKeySequence("Ctrl+Alt+S"))
        fm.addSeparator()
        self._act(fm, "Quit", self.close, QKeySequence.StandardKey.Quit)
        em = mb.addMenu("&Edit")
        undo = self.params.undo_stack.createUndoAction(self, "Undo")
        undo.setShortcut(QKeySequence.StandardKey.Undo)
        redo = self.params.undo_stack.createRedoAction(self, "Redo")
        redo.setShortcut(QKeySequence.StandardKey.Redo)
        em.addAction(undo)
        em.addAction(redo)
        em.addSeparator()
        self._act(em, "Revert All Edits", self.revert_all)
        self._act(em, "Elements to Edit…", self.choose_subsystem)
        vm = mb.addMenu("&Diagram")
        self._act(vm, "Redraw", self.redraw, QKeySequence("F5"))
        a = self._act(vm, "Redraw ", self.redraw, QKeySequence("Ctrl+R"))
        a.setVisible(False)
        self._act(vm, "Cancel Mapping", self.runner.cancel, QKeySequence("Esc"))
        self._act(vm, "Reset View", self.reset_view, QKeySequence("Ctrl+0"))
        self._act(vm, "Export Figure…", self.export_figure)
        hm = mb.addMenu("&Help")
        self._act(hm, "About", lambda: QMessageBox.about(
            self, "tdb-tweak", f"tdb-tweak {__version__}\n\nHand-tune TDB parameters against a pycalphad map.\n"
                               f"Column generation available: {'yes' if cg_available() else 'no'}"))

    def _act(self, menu, text, slot, shortcut=None):
        a = QAction(text, self)
        if shortcut is not None:
            a.setShortcut(shortcut)
        a.triggered.connect(slot)
        menu.addAction(a)
        return a

    # ===================================================================================== documents
    def open_tdb_dialog(self):
        start = self.settings.value("last_dir", os.getcwd())
        path, _ = QFileDialog.getOpenFileName(self, "Open TDB", start, "TDB files (*.tdb *.TDB);;All files (*)")
        if path:
            self.open_tdb(path)

    def open_tdb(self, path: str, subsystem: Optional[List[str]] = None) -> bool:
        if not self._confirm_discard():
            return False
        try:
            doc = TDBDocument.from_file(path)
            from pycalphad import Database
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                phase_db = Database.from_string(doc.compute_text(), fmt="tdb")
        except Exception as ex:
            QMessageBox.critical(self, "Open TDB", f"Could not read {path}:\n{ex}")
            return False
        self.settings.setValue("last_dir", os.path.dirname(path))
        self.doc, self.phase_db = doc, phase_db
        self.result = None
        self.cache.clear()
        self._queue = []
        self.excluded_phases = set()
        els = doc.element_list()
        self.subsystem = [e for e in (subsystem or els) if e in els]
        for w in doc.warnings:
            self._log(f"warning: {w}")
        self._log(f"opened {path}: {len(doc.rows)} editable numbers, {len(doc.vvdefs)} VV symbols")
        self.params.set_document(doc, self.subsystem)
        self._fill_element_combos()
        self._update_title()
        self.stale = True
        self._update_state_label()
        if subsystem is None and len(els) > 3:
            QTimer.singleShot(0, self.choose_subsystem)
        self.replot()
        return True

    def _confirm_discard(self) -> bool:
        if self.doc is None or self.doc.modified_count() == 0:
            return True
        r = QMessageBox.question(self, "Unsaved edits",
                                 f"{self.doc.modified_count()} values were changed and not saved. Discard them?",
                                 QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Save
                                 | QMessageBox.StandardButton.Cancel)
        if r == QMessageBox.StandardButton.Save:
            return self.save_tdb()
        return r == QMessageBox.StandardButton.Discard

    def save_tdb(self) -> bool:
        if self.doc is None:
            return False
        if not self.doc.path:
            return self.save_tdb_as()
        return self._save_to(self.doc.path)

    def save_tdb_as(self) -> bool:
        if self.doc is None:
            return False
        path, _ = QFileDialog.getSaveFileName(self, "Save TDB as", self.doc.path or "", "TDB files (*.tdb)")
        return self._save_to(path) if path else False

    def _save_to(self, path: str) -> bool:
        try:
            n = self.doc.modified_count()
            self.doc.save(path)
            self.doc.path = path
            self.doc.mark_saved()
        except Exception as ex:
            QMessageBox.critical(self, "Save TDB", str(ex))
            return False
        self.params.rebuild()
        self._update_title()
        self._log(f"saved {n} changed values to {path}")
        self.statusBar().showMessage(f"Saved {path}", 5000)
        return True

    def revert_all(self):
        self.params.revert_all()

    def _update_title(self):
        name = os.path.basename(self.doc.path) if self.doc and self.doc.path else "untitled"
        mod = f"  ({self.doc.modified_count()} changed)" if self.doc and self.doc.modified_count() else ""
        self.setWindowTitle(f"tdb-tweak — {name}{mod}")

    # ===================================================================================== subsystem / phases
    def choose_subsystem(self):
        if self.doc is None:
            return
        dlg = CheckListDialog("Elements to edit", self.doc.element_list(), set(self.subsystem), self, 2)
        if dlg.exec():
            self.subsystem = dlg.checked()
            self.params.set_elements(self.subsystem)
            self._fill_element_combos()

    def _fill_element_combos(self):
        els = self.subsystem
        cur = [c.currentText() for c in self.el]
        for i, c in enumerate(self.el):
            c.blockSignals(True)
            c.clear()
            c.addItems(els)
            if cur[i] in els:
                c.setCurrentText(cur[i])
            elif i < len(els):
                c.setCurrentIndex(i)
            c.blockSignals(False)
        if len(els) == 2 and self.kind.currentData() != "binary":
            self.kind.setCurrentIndex(0)
        self.kind.model().item(1).setEnabled(len(els) >= 3)
        self.kind.model().item(2).setEnabled(len(els) >= 3)
        self._on_elements_changed()

    def diagram_elements(self) -> Tuple[str, ...]:
        n = 2 if self.kind.currentData() == "binary" else 3
        return tuple(c.currentText() for c in self.el[:n])

    def available_phases(self, elements=None) -> List[str]:
        if self.phase_db is None:
            return []
        from pycalphad.core.utils import filter_phases, unpack_species
        comps = sorted(set(elements or self.diagram_elements()) | {"VA"})
        return sorted(filter_phases(self.phase_db, unpack_species(self.phase_db, comps), list(self.phase_db.phases)))

    def choose_phases(self):
        ph = self.available_phases()
        if not ph:
            return
        dlg = CheckListDialog("Phases in the diagram", ph, set(ph) - self.excluded_phases, self, 1)
        if dlg.exec():
            chosen = set(dlg.checked())
            self.excluded_phases = (self.excluded_phases - set(ph)) | (set(ph) - chosen)
            self.stale = True
            self._update_state_label()

    # ===================================================================================== diagram controls
    def _update_kind_widgets(self):
        k = self.kind.currentData()
        rng_T = k in ("binary", "isopleth")
        for w in (self.lbl_T, self.tmin, self.lbl_Tto, self.tmax, self.lbl_Tstep, self.tstep):
            w.setVisible(rng_T)
        for w in (self.lbl_Tsec, self.tsec):
            w.setVisible(k == "ternary")
        for w in (self.lbl_y, self.ymin, self.lbl_yto, self.ymax):
            w.setVisible(k == "ternary")
        for w in (self.lbl_fixed, self.fixed):
            w.setVisible(k == "isopleth")
        for w in (self.el[2], self.el_label[2]):
            w.setVisible(k != "binary")
        self.el_label[0].setText("A" if k != "ternary" else "Corners:")
        self.el_label[1].setText("x:" if k != "ternary" else "")
        self.el_label[2].setText("fixed:" if k == "isopleth" else "")
        self.tielines.setChecked(k == "ternary")
        self._update_range_labels()

    def _update_range_labels(self):
        els = self.diagram_elements() + ("?", "?", "?")
        B = els[1].capitalize()
        self.lbl_x.setText(f"x({B}) from")
        self.lbl_y.setText(f"x({els[2].capitalize()}) from")
        self.lbl_fixed.setText(f"x({els[2].capitalize()}) =")
        self.lbl_xstep.setText("step")
        tu = self.t_unit.currentText()
        for s in (self.tmin, self.tmax, self.tsec):
            s.setSuffix(f" {tu}")
        self.tstep.setSuffix(f" {'K' if tu == 'K' else tu}")

    def _on_kind_changed(self, *a):
        self._update_kind_widgets()
        self._on_elements_changed()

    def _on_elements_changed(self, *a):
        self._update_range_labels()
        self._refresh_zpf_temps()
        self.stale = True
        self._update_state_label()
        self.replot()

    def _on_t_unit_changed(self, new: str):
        old = self._t_unit_prev
        self._t_unit_prev = new
        for s in (self.tmin, self.tmax, self.tsec):
            s.blockSignals(True)
            s.setValue(float(U.t_from_K(U.t_to_K(s.value(), old), new)))
            s.blockSignals(False)
        if (old == "°F") != (new == "°F"):
            f = 9 / 5 if new == "°F" else 5 / 9
            self.tstep.setValue(self.tstep.value() * f)
        self._update_range_labels()
        self.replot()

    def _T_K(self, spin) -> float:
        return float(U.t_to_K(spin.value(), self.t_unit.currentText()))

    def _on_section_T_changed(self):
        if self.kind.currentData() == "ternary":
            self.stale = True
            self._update_state_label()
            self.replot()

    def diagram_settings(self) -> dict:
        """Everything that defines the current diagram (temperatures in K, compositions in at.%)."""
        return {
            "kind": self.kind.currentData(), "elements": list(self.diagram_elements()),
            "t_unit": self.t_unit.currentText(), "x_unit": self.x_unit.currentText(),
            "tmin_K": self._T_K(self.tmin), "tmax_K": self._T_K(self.tmax),
            "tstep": self.tstep.value() * (5 / 9 if self.t_unit.currentText() == "°F" else 1.0),
            "tsec_K": self._T_K(self.tsec), "xmin": self.xmin.value(), "xmax": self.xmax.value(),
            "xstep": self.xstep.value(), "ymin": self.ymin.value(), "ymax": self.ymax.value(),
            "fixed": self.fixed.value(), "pdens": self.pdens.value(), "cg": bool(self.solver.currentData()),
            "tielines": self.tielines.isChecked(),
        }

    def apply_diagram_settings(self, g: dict, zoom=None):
        """Set every diagram control from ``g`` (one redraw at the end, no intermediate maps)."""
        widgets = [self.kind, *self.el, self.t_unit, self.x_unit, self.tmin, self.tmax, self.tstep, self.tsec,
                   self.xmin, self.xmax, self.xstep, self.ymin, self.ymax, self.fixed, self.pdens, self.solver,
                   self.tielines]
        for w in widgets:
            w.blockSignals(True)
        try:
            self.kind.setCurrentIndex([k for k, _ in KINDS].index(g.get("kind", "binary")))
            self._update_kind_widgets()
            for c, e in zip(self.el, g.get("elements", [])):
                c.setCurrentText(e)
            self.t_unit.setCurrentText(g.get("t_unit", "K"))
            self._t_unit_prev = self.t_unit.currentText()
            self.x_unit.setCurrentText(g.get("x_unit", "at.%"))
            tu = self.t_unit.currentText()
            self.tmin.setValue(float(U.t_from_K(g.get("tmin_K", 300), tu)))
            self.tmax.setValue(float(U.t_from_K(g.get("tmax_K", 2000), tu)))
            self.tsec.setValue(float(U.t_from_K(g.get("tsec_K", 1000), tu)))
            self.tstep.setValue(g.get("tstep", 10) * (9 / 5 if tu == "°F" else 1.0))
            for name in ("xmin", "xmax", "xstep", "ymin", "ymax", "fixed"):
                if name in g:
                    getattr(self, name).setValue(g[name])
            self.pdens.setValue(g.get("pdens", 500))
            self.solver.setCurrentIndex(1 if g.get("cg") and cg_available() else 0)
            self.tielines.setChecked(g.get("tielines", g.get("kind") == "ternary"))
        finally:
            for w in widgets:
                w.blockSignals(False)
        self._update_range_labels()
        self._refresh_zpf_temps()
        self._pending_zoom = ("set", zoom)
        self.replot()

    def request_from_settings(self, g: dict) -> Tuple[Optional[MapRequest], str]:
        k = g["kind"]
        els = tuple(g["elements"][: 2 if k == "binary" else 3])
        if len(set(els)) != len(els):
            return None, "Choose different elements."
        missing = [e for e in els if e not in self.subsystem]
        if missing:
            return None, f"Elements {missing} are not in the edited set."
        if k in ("binary", "isopleth") and g["tmax_K"] <= g["tmin_K"]:
            return None, "The upper temperature must be above the lower one."
        if g["xmax"] <= g["xmin"] or (k == "ternary" and g["ymax"] <= g["ymin"]):
            return None, "The upper composition must be above the lower one."
        allowed = [p for p in self.available_phases(els) if p not in self.excluded_phases]
        req = MapRequest(kind=k, elements=els, phases=tuple(allowed) or None,
                         tmin=g["tmin_K"], tmax=g["tmax_K"], tstep=g["tstep"], temperature=g["tsec_K"],
                         xmin=g["xmin"] / 100, xmax=g["xmax"] / 100, xstep=g["xstep"] / 100,
                         ymin=g["ymin"] / 100, ymax=g["ymax"] / 100, fixed=g["fixed"] / 100,
                         use_cg=bool(g.get("cg")), global_min_pdens=int(g.get("pdens", 500)))
        if k == "isopleth" and req.xmax + req.fixed > 1:
            req.xmax = 1 - req.fixed
        return req, ""

    def build_request(self) -> Optional[MapRequest]:
        return self.request_from_settings(self.diagram_settings())[0]

    def current_view(self) -> View:
        k = self.kind.currentData()
        return View(kind=k, elements=self.diagram_elements(), t_unit=self.t_unit.currentText(),
                    x_unit=self.x_unit.currentText(), masses=self.doc.masses() if self.doc else {},
                    temperature=self._T_K(self.tsec), fixed=self.fixed.value() / 100,
                    zpf_tolerance=self.zpf_tol.value(), tieline_every=3 if self.tielines.isChecked() else 0)

    # ===================================================================================== mapping
    def _stamp(self):
        return values_stamp(self.doc.compute_text(), self.doc.parameter_values())

    def _current_key(self):
        req = self.build_request()
        return request_key(req) if req is not None else None

    def redraw(self):
        if self.doc is None:
            return
        req, err = self.request_from_settings(self.diagram_settings())
        if req is None:
            QMessageBox.warning(self, "Diagram", err)
            return
        self._queue = []
        self._map(req, "")

    def _map(self, req: MapRequest, label: str, from_queue: bool = False) -> bool:
        """Show the cached map for ``req`` with the current values, or start mapping it.

        Returns True when a mapping job was started."""
        key, stamp = request_key(req), self._stamp()
        hit = self.cache.get(key, stamp)
        if hit is not None:
            if key == self._current_key():
                self.result = hit
                self._update_state_label(done=f"up to date ({hit.n_lines} lines; mapped in {hit.seconds:.2f} s)")
                self.replot()
            return False
        if self.runner.running and self._pending is not None:
            if self._pending[0] == key and self._pending[1] == stamp:
                return True                                  # already mapping exactly this
            if self._pending[4]:                             # interrupting "Map all": do that one later
                self._queue.insert(0, (self._pending[3], self._pending[2]))
        self._queue = [(r, l) for r, l in self._queue if request_key(r) != key]
        self._pending = (key, stamp, label, req, from_queue)
        self._map_started = time.perf_counter()
        self.runner.submit(self.doc.compute_text(), list(self.doc.parameter_values().keys()),
                           self.doc.parameter_values(), req)
        self.cancel_btn.setEnabled(True)
        self.state_lbl.setText(f"<span style='color:#0a58ca'>● mapping{' ' + label if label else ''}…</span>")
        self._refresh_view_buttons()
        return True

    def _next_in_queue(self):
        while self._queue:
            req, label = self._queue.pop(0)
            n_left = len(self._queue)
            if self._map(req, f"{label} ({n_left} more queued)" if n_left else label, from_queue=True):
                return

    def _on_progress(self, res: MapResult):
        label = self._pending[2] if self._pending else ""
        self.state_lbl.setText(f"<span style='color:#0a58ca'>● mapping{' ' + label if label else ''}… "
                               f"{res.n_lines} lines, {time.perf_counter() - self._map_started:.1f} s</span>")
        if request_key(res.request) == self._current_key():
            self.result = res
            self.replot()

    def _on_finished(self, res: MapResult):
        self.cancel_btn.setEnabled(False)
        key = request_key(res.request)
        if self._pending is not None and self._pending[0] == key:
            self.cache.put(key, self._pending[1], res)
        self._pending = None
        msg = f"{res.n_lines} lines, {len(res.invariants)} invariants in {res.seconds:.2f} s"
        if res.message:
            msg += f"  ({res.message})"
        self._log(f"map {'-'.join(res.request.elements)} ({res.request.kind}"
                  f"{', cg' if res.request.use_cg else ''}): {msg}")
        if key == self._current_key():
            self.result = res
            self.replot()
        self._update_state_label(done=msg)
        self._next_in_queue()

    def _on_failed(self, msg: str):
        self.cancel_btn.setEnabled(False)
        self._pending = None
        self._log("mapping failed: " + msg)
        self.state_lbl.setText("<span style='color:#b00'>● mapping failed (see Log)</span>")
        self.tabs.setCurrentWidget(self.log)
        self._next_in_queue()

    def _on_cancelled(self):
        self.cancel_btn.setEnabled(False)
        self._pending = None
        self._queue = []
        self.state_lbl.setText("<span style='color:#888'>● cancelled</span>")
        self._refresh_view_buttons()

    def _on_values_changed(self, needs_rebuild: bool):
        self.stale = True
        if needs_rebuild:
            self._log("a coefficient became editable: the next map recompiles the models once")
        self._update_state_label()
        self._update_title()

    def _on_committed(self):
        if self.auto_cb.isChecked():
            QTimer.singleShot(0, self.redraw)

    def _update_state_label(self, done: Optional[str] = None):
        self._sync_active_view()
        self._refresh_view_buttons()
        if self.runner.running or self.doc is None:
            return
        key = self._current_key()
        if key is not None and self.cache.get(key, self._stamp()) is not None:
            self.state_lbl.setText(f"<span style='color:#198754'>● {done or 'up to date'}</span>")
        else:
            self.state_lbl.setText("<span style='color:#c77700'>● not mapped with these settings/values — F5</span>")

    # ===================================================================================== plotting
    def _visible_images(self, view: View):
        out, skipped = [], []
        for e in self.images:
            c = e.cal
            if not c.visible or c.kind != view.kind or tuple(c.elements) != tuple(view.elements):
                continue
            if c.kind != "ternary" and c.t_unit != view.t_unit:
                skipped.append(os.path.basename(c.image_path))
                continue
            if U.is_mass(c.x_unit) != U.is_mass(view.x_unit) or \
                    (c.kind != "ternary" and U.scale(c.x_unit) != U.scale(view.x_unit)):
                skipped.append(os.path.basename(c.image_path))
                continue
            try:
                arr, extent = e.rectified()
            except Exception as ex:
                self._log(f"image {c.image_path}: {ex}")
                continue
            out.append((arr, extent, c.alpha))
        if skipped:
            self.statusBar().showMessage("Not shown (units differ from the diagram's): " + ", ".join(skipped), 8000)
        return out

    def replot(self):
        if self.doc is None:
            self.figure.clear()
            self.canvas.draw_idle()
            return
        view = self.current_view()
        res = self._result_for_display(view)
        key = (view.kind, view.elements, view.t_unit, view.x_unit, view.temperature if view.kind == "ternary" else 0)
        if self._pending_zoom is not None:                 # a saved view was just applied
            self._zoom = self._pending_zoom[1]
            self._pending_zoom = None
        elif key != self._view_key:
            self._zoom = None
        else:
            cur = self._axes_limits()
            if cur is not None and self._last_drawn is not None and cur != self._last_drawn:
                self._zoom = cur                           # the user zoomed or panned since the last draw
        limits = self._zoom
        images = self._visible_images(view)
        try:
            ax, self._zpf_artists = draw_all(self.figure, view, res, self.zpf, images, self.colors, limits)
        except Exception as ex:
            self._log(f"drawing failed: {ex}")
            return
        if limits is None:
            if images and view.kind != "ternary":
                x0 = min(i[1][0] for i in images)
                x1 = max(i[1][1] for i in images)
                y0 = min(i[1][2] for i in images)
                y1 = max(i[1][3] for i in images)
                ax.set_xlim(x0, x1)
                ax.set_ylim(y0, y1)
        self._last_drawn = (tuple(ax.get_xlim()), tuple(ax.get_ylim()))
        self._view_key = key
        self.toolbar.update()
        self.canvas.draw_idle()

    def _axes_limits(self):
        if not self.figure.axes:
            return None
        ax = self.figure.axes[0]
        return (tuple(ax.get_xlim()), tuple(ax.get_ylim()))

    def _result_for_display(self, view: View) -> Optional[MapResult]:
        """The up-to-date map of the current settings if cached, else the newest map of them (old values),
        else the last map of the same diagram type and elements."""
        if self.doc is not None:
            key = self._current_key()
            if key is not None:
                hit = self.cache.get(key, self._stamp())
                if hit is not None:
                    return hit
                if self.result is not None and request_key(self.result.request) == key:
                    return self.result
                old = self.cache.latest_for(key)
                if old is not None:
                    return old
        res = self.result
        if res is not None and (res.request.kind != view.kind or tuple(res.request.elements) != view.elements):
            return None
        if res is not None and view.kind == "ternary" and abs(res.request.temperature - view.temperature) > 1e-6:
            return None
        return res

    def reset_view(self):
        self._zoom = None
        self._last_drawn = None
        self._view_key = None
        self.figure.clear()
        self.replot()

    # ===================================================================================== saved views
    def _build_views_bar(self) -> QToolBar:
        bar = QToolBar()
        bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        bar.setStyleSheet(
            "QToolBar { border: 0; spacing: 4px; }"
            "QToolButton { padding: 2px 9px; border: 1px solid #b8b8b8; border-radius: 4px; }"
            "QToolButton:checked { background: #0a58ca; color: white; border-color: #0a58ca; }")
        self.views_bar = bar
        self._view_buttons: List[QToolButton] = []
        for i in range(9):
            QShortcut(QKeySequence(f"Ctrl+{i + 1}"), self, activated=lambda i=i: self.apply_view(i))
        self._rebuild_views_bar()
        return bar

    def _rebuild_views_bar(self):
        bar = self.views_bar
        bar.clear()
        self._view_buttons = []
        bar.addWidget(QLabel(" Views: "))
        if not self.views:
            hint = QLabel("none yet — set up a diagram you will come back to and press “+ Save view”")
            hint.setStyleSheet("color: #888;")
            bar.addWidget(hint)
        for i, v in enumerate(self.views):
            b = QToolButton()
            b.setText(v.name)
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, i=i: self.apply_view(i))
            b.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            b.customContextMenuRequested.connect(lambda pos, i=i, b=b: self._view_menu(i, b.mapToGlobal(pos)))
            bar.addWidget(b)
            self._view_buttons.append(b)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        bar.addWidget(spacer)
        add = QToolButton()
        add.setText("+ Save view")
        add.setToolTip("Save the current diagram (type, elements, ranges, section, units, solver and zoom) as a button")
        add.clicked.connect(self.save_view)
        bar.addWidget(add)
        mall = QToolButton()
        mall.setText("Map all views")
        mall.setToolTip("Map every saved view with the current values, one after another in the background,\n"
                        "so switching between them is instant (views not mapped with the current values are italic)")
        mall.clicked.connect(self.map_all_views)
        bar.addWidget(mall)
        self._refresh_view_buttons()

    def _refresh_view_buttons(self):
        if not getattr(self, "_view_buttons", None):
            return
        stamp = self._stamp() if self.doc is not None else None
        running = self._pending[0] if (self._pending is not None and self.runner.running) else None
        for i, (b, v) in enumerate(zip(self._view_buttons, self.views)):
            b.setChecked(i == self._active_view)
            req = self.request_from_settings(v.settings)[0] if self.doc is not None else None
            k = request_key(req) if req is not None else None
            fresh = k is not None and self.cache.get(k, stamp) is not None
            f = QFont(b.font())
            f.setItalic(not fresh)
            b.setFont(f)
            state = "map up to date" if fresh else ("mapping now…" if k is not None and k == running
                                                    else "not mapped with the current values")
            short = f"Ctrl+{i + 1}" if i < 9 else ""
            b.setToolTip(f"{v.name}\n{state}\n{short}  ·  right-click: update, rename, move, delete")

    def apply_view(self, i: int):
        if self.doc is None or not (0 <= i < len(self.views)):
            return
        v = self.views[i]
        req, err = self.request_from_settings(v.settings)
        if req is None:
            self.statusBar().showMessage(f"View “{v.name}”: {err}", 8000)
            return
        self._active_view = i
        self.apply_diagram_settings(v.settings, zoom=[tuple(v.limits[0]), tuple(v.limits[1])]
                                    if v.limits else None)
        self._map(req, v.name)
        self._update_state_label()

    def _current_zoom(self):
        cur = self._axes_limits()
        if cur is not None and self._last_drawn is not None and cur != self._last_drawn:
            return cur
        return self._zoom

    def save_view(self):
        if self.doc is None:
            return
        g = self.diagram_settings()
        zoom = self._current_zoom()
        name, ok = QInputDialog.getText(self, "Save view", "Button name:", text=auto_name(g, zoom is not None))
        if not ok or not name.strip():
            return
        self.views.append(SavedView(name.strip(), g, [list(zoom[0]), list(zoom[1])] if zoom else None))
        self._active_view = len(self.views) - 1
        self._rebuild_views_bar()

    def _view_menu(self, i: int, pos):
        m = QMenu(self)
        upd = m.addAction("Update to the current diagram and zoom")
        ren = m.addAction("Rename…")
        m.addSeparator()
        left = m.addAction("Move left")
        right = m.addAction("Move right")
        left.setEnabled(i > 0)
        right.setEnabled(i < len(self.views) - 1)
        m.addSeparator()
        dele = m.addAction("Delete")
        a = m.exec(pos)
        if a is None:
            return
        v = self.views[i]
        if a == upd:
            zoom = self._current_zoom()
            v.settings = self.diagram_settings()
            v.limits = [list(zoom[0]), list(zoom[1])] if zoom else None
            self._active_view = i
        elif a == ren:
            name, ok = QInputDialog.getText(self, "Rename view", "Button name:", text=v.name)
            if ok and name.strip():
                v.name = name.strip()
        elif a in (left, right):
            j = i - 1 if a == left else i + 1
            self.views[i], self.views[j] = self.views[j], self.views[i]
            if self._active_view in (i, j):
                self._active_view = j if self._active_view == i else i
        elif a == dele:
            del self.views[i]
            if self._active_view == i:
                self._active_view = None
            elif self._active_view is not None and self._active_view > i:
                self._active_view -= 1
        self._rebuild_views_bar()

    def _sync_active_view(self):
        """Un-highlight the active view once the diagram settings no longer match it."""
        if self._active_view is None or self.doc is None:
            return
        if self._active_view >= len(self.views) or \
                not same_diagram(self.views[self._active_view].settings, self.diagram_settings()):
            self._active_view = None

    def map_all_views(self):
        if self.doc is None or not self.views:
            return
        order = list(range(len(self.views)))
        if self._active_view is not None:
            order.remove(self._active_view)
            order.insert(0, self._active_view)
        stamp = self._stamp()
        jobs = []
        for i in order:
            req, err = self.request_from_settings(self.views[i].settings)
            if req is None:
                self._log(f"view “{self.views[i].name}” skipped: {err}")
            elif self.cache.get(request_key(req), stamp) is None:
                jobs.append((req, self.views[i].name))
        if not jobs:
            self.statusBar().showMessage("Every saved view is already mapped with the current values.", 5000)
            return
        self._queue = jobs
        if not self.runner.running:
            self._next_in_queue()

    def _on_pick(self, ev):
        art = ev.artist
        info = getattr(art, "zpf_info", None)
        if not info or not len(ev.ind):
            return
        i = int(ev.ind[0])
        self.statusBar().showMessage(info[i], 10000)
        self._picked_T = art.zpf_T[i]
        self.map_picked_btn.setEnabled(self.kind.currentData() == "ternary")
        self.map_picked_btn.setText(f"Map section at {self._picked_T:.6g} K (picked point)")

    def export_figure(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export figure", "", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            self.figure.savefig(path, dpi=200, bbox_inches="tight")

    # ===================================================================================== images
    def add_image(self):
        if self.doc is None:
            QMessageBox.information(self, "Images", "Open a TDB first (the elements come from it).")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Add literature image", self.settings.value("img_dir", ""),
                                              "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.gif)")
        if not path:
            return
        self.settings.setValue("img_dir", os.path.dirname(path))
        cal = Calibration.load_sidecar(path)
        if cal is not None and QMessageBox.question(
                self, "Calibration found", "This image has a saved calibration. Use it?") != QMessageBox.StandardButton.Yes:
            cal = None
        if cal is None:
            kind = self.kind.currentData()
            cal = Calibration(path, kind=kind, t_unit=self.t_unit.currentText(), x_unit=self.x_unit.currentText())
            cal.elements = list(self.diagram_elements())
            if kind == "ternary":
                cal.mode = "points"
                full = 100.0 if cal.x_unit.endswith("%") else 1.0
                cal.bounds = [0.0, full, 0.0, full]
            else:
                full = 100.0 if cal.x_unit.endswith("%") else 1.0
                cal.bounds = [0.0, full, float(self.tmin.value()), float(self.tmax.value())]
            cal.default_points()
            dlg = CalibrationDialog(path, self.subsystem, cal, parent=self)
            if not dlg.exec():
                return
            cal = dlg.cal
        cal.image_path = os.path.abspath(cal.image_path)
        self.images.append(ImageEntry(cal))
        self._refresh_image_list(select=len(self.images) - 1)
        self._apply_image_setup(cal)

    def calibrate_image(self):
        i = self.img_list.currentRow()
        if i < 0 or self.doc is None:
            return
        e = self.images[i]
        import copy
        dlg = CalibrationDialog(e.cal.image_path, sorted(set(self.subsystem) | set(e.cal.elements)),
                                copy.deepcopy(e.cal), parent=self)
        if dlg.exec():
            e.cal = dlg.cal
            self._refresh_image_list(select=i)
            self._apply_image_setup(e.cal)

    def remove_image(self):
        i = self.img_list.currentRow()
        if i >= 0:
            del self.images[i]
            self._refresh_image_list()
            self.replot()

    def _refresh_image_list(self, select: Optional[int] = None):
        self.img_list.blockSignals(True)
        self.img_list.clear()
        for e in self.images:
            c = e.cal
            label = f"{os.path.basename(c.image_path)}  —  {'-'.join(x.capitalize() for x in c.elements)} " \
                    f"{c.kind}, {c.x_unit}" + ("" if c.kind == "ternary" else f", {c.t_unit}")
            it = QListWidgetItem(label)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if c.visible else Qt.CheckState.Unchecked)
            it.setToolTip(c.image_path)
            self.img_list.addItem(it)
        self.img_list.blockSignals(False)
        if select is not None:
            self.img_list.setCurrentRow(select)

    def _on_image_toggled(self, item: QListWidgetItem):
        i = self.img_list.row(item)
        cal = self.images[i].cal
        cal.visible = item.checkState() == Qt.CheckState.Checked
        if cal.visible:
            self._apply_image_setup(cal)
        else:
            self.replot()

    def _on_image_selected(self, i: int):
        if 0 <= i < len(self.images):
            self.alpha.blockSignals(True)
            self.alpha.setValue(int(self.images[i].cal.alpha * 100))
            self.alpha.blockSignals(False)

    def _on_alpha(self, v: int):
        i = self.img_list.currentRow()
        if 0 <= i < len(self.images):
            self.images[i].cal.alpha = v / 100
            self.replot()

    def _apply_image_setup(self, cal: Calibration):
        """Switch the diagram to the image's type, elements and units."""
        missing = [e for e in cal.elements if e not in self.subsystem]
        if missing:
            self.statusBar().showMessage(f"The image needs elements {missing} that are not in the edited set", 8000)
            self.replot()
            return
        self.kind.blockSignals(True)
        self.kind.setCurrentIndex([k for k, _ in KINDS].index(cal.kind))
        self.kind.blockSignals(False)
        self._update_kind_widgets()
        for c, e in zip(self.el, cal.elements):
            c.blockSignals(True)
            c.setCurrentText(e)
            c.blockSignals(False)
        if cal.kind != "ternary":
            self.t_unit.setCurrentText(cal.t_unit)
        self.x_unit.setCurrentText(cal.x_unit)
        self._view_key = None
        self._on_elements_changed()

    def fit_ranges_to_image(self):
        view = self.current_view()
        cands = [e.cal for e in self.images if e.cal.visible and e.cal.kind == view.kind
                 and tuple(e.cal.elements) == view.elements]
        if not cands or self.doc is None:
            QMessageBox.information(self, "Fit ranges", "No visible image for this diagram.")
            return
        c = cands[0]
        b0, b1, c0, c1 = c.bounds
        masses = self.doc.masses()
        if c.kind in ("binary", "isopleth"):
            A, B = c.elements[0], c.elements[1]
            if c.kind == "binary":
                xs = U.binary_x_from_display(np.array([b0, b1]), A, B, masses, c.x_unit)
            else:
                xs = np.array([b0, b1]) / U.scale(c.x_unit)
            self.xmin.setValue(max(0.0, float(np.min(xs)) * 100))
            self.xmax.setValue(min(100.0, float(np.max(xs)) * 100))
            tu = self.t_unit.currentText()
            self.tmin.setValue(float(U.t_from_K(U.t_to_K(min(c0, c1), c.t_unit), tu)))
            self.tmax.setValue(float(U.t_from_K(U.t_to_K(max(c0, c1), c.t_unit), tu)))
        else:
            A, B, C = c.elements
            s = U.scale(c.x_unit)
            corners = [(b, cc) for b in (b0, b1) for cc in (c0, c1) if b + cc <= s + 1e-9]
            fr = [U.x_from_display({B: b, C: cc}, [A, B, C], masses, c.x_unit) for b, cc in corners]
            self.xmin.setValue(max(0.0, min(f[B] for f in fr) * 100))
            self.xmax.setValue(min(100.0, max(f[B] for f in fr) * 100))
            self.ymin.setValue(max(0.0, min(f[C] for f in fr) * 100))
            self.ymax.setValue(min(100.0, max(f[C] for f in fr) * 100))
        self.stale = True
        self._update_state_label()

    # ===================================================================================== ZPF
    def add_zpf_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "ZPF datasets", self.settings.value("zpf_dir", ""),
                                                "ESPEI datasets (*.json)")
        if paths:
            self.settings.setValue("zpf_dir", os.path.dirname(paths[0]))
            self._add_zpf(paths)

    def add_zpf_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Folder of ZPF datasets", self.settings.value("zpf_dir", ""))
        if d:
            self.settings.setValue("zpf_dir", d)
            self._add_zpf([d])

    def _add_zpf(self, paths: List[str]):
        ds, msgs = load_paths(paths)
        known = {d.path for d in self.zpf}
        new = [d for d in ds if d.path not in known]
        self.zpf += new
        for m in msgs:
            self._log(f"ZPF: {m}")
        self._log(f"ZPF: loaded {len(new)} datasets")
        self._refresh_zpf_list()
        self.replot()

    def remove_zpf(self):
        i = self.zpf_list.currentRow()
        if i >= 0:
            del self.zpf[i]
            self._refresh_zpf_list()
            self.replot()

    def clear_zpf(self):
        self.zpf = []
        self._refresh_zpf_list()
        self.replot()

    def _refresh_zpf_list(self):
        self.zpf_list.blockSignals(True)
        self.zpf_list.clear()
        for d in self.zpf:
            n = len(d.tielines)
            it = QListWidgetItem(f"{d.label}  [{'-'.join(sorted(d.elements))}, {n} tie-lines]")
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if d.enabled else Qt.CheckState.Unchecked)
            it.setToolTip(f"{d.path}\nphases: {', '.join(d.phases)}\n{d.comment}")
            self.zpf_list.addItem(it)
        self.zpf_list.blockSignals(False)
        self._refresh_zpf_temps()

    def _on_zpf_toggled(self, item: QListWidgetItem):
        self.zpf[self.zpf_list.row(item)].enabled = item.checkState() == Qt.CheckState.Checked
        self._refresh_zpf_temps()
        self.replot()

    def _refresh_zpf_temps(self):
        self.zpf_temps.clear()
        if self.kind.currentData() != "ternary":
            self.zpf_temps.addItem("(for ternary isothermal sections)")
            return
        tu = self.t_unit.currentText()
        for T, n in temperatures(self.zpf, self.diagram_elements()):
            it = QListWidgetItem(f"{T:g} K  ({U.t_from_K(T, tu):.6g} {tu})  —  {n} tie-lines")
            it.setData(Qt.ItemDataRole.UserRole, T)
            self.zpf_temps.addItem(it)

    def _on_zpf_temp_activated(self, item: QListWidgetItem):
        T = item.data(Qt.ItemDataRole.UserRole)
        if T is not None:
            self._map_section_at(float(T))

    def _map_at_picked(self):
        if self._picked_T is not None:
            self._map_section_at(self._picked_T)

    def _map_section_at(self, T: float):
        self.tsec.setValue(float(U.t_from_K(T, self.t_unit.currentText())))
        self.redraw()

    # ===================================================================================== projects
    def project_dict(self) -> dict:
        d = {
            "version": 1,
            "tdb": self.doc.path if self.doc else None,
            "subsystem": self.subsystem,
            "excluded_phases": sorted(self.excluded_phases),
            "diagram": self.diagram_settings(),
            "views": [v.to_dict() for v in self.views],
            "images": [e.cal.to_dict() for e in self.images],
            "zpf": [{"path": z.path, "enabled": z.enabled} for z in self.zpf],
            "zpf_tolerance": self.zpf_tol.value(),
            "table": {"sort": self.params.sort_combo.currentIndex(), "nonvv": self.params.show_nonvv.isChecked(),
                      "lower": self.params.show_lower.isChecked()},
        }
        if self.doc is not None:
            d["unsaved_edits"] = {
                "vv": {k: v for k, v in self.doc.vv_values.items() if v != self.doc.vv_orig[k]},
                "coef": {str(k): v for k, v in self.doc.coef_values.items() if v != self.doc.coef_orig[k]},
            }
        return d

    def save_project_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save project", self.settings.value("last_dir", ""),
                                              "tdb-tweak project (*.tdbtweak)")
        if path:
            if not path.endswith(".tdbtweak"):
                path += ".tdbtweak"
            with open(path, "w") as f:
                json.dump(self.project_dict(), f, indent=1)
            self._log(f"project saved to {path}")

    def open_project_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open project", self.settings.value("last_dir", ""),
                                              "tdb-tweak project (*.tdbtweak)")
        if path:
            self.open_project(path)

    def open_project(self, path: str):
        with open(path) as f:
            d = json.load(f)
        base = os.path.dirname(os.path.abspath(path))

        def res(p):
            return p if (p is None or os.path.isabs(p)) else os.path.join(base, p)

        if d.get("tdb") and not self.open_tdb(res(d["tdb"]), d.get("subsystem")):
            return
        self.excluded_phases = set(d.get("excluded_phases", []))
        t = d.get("table", {})
        self.params.sort_combo.setCurrentIndex(t.get("sort", 1))
        self.params.show_nonvv.setChecked(t.get("nonvv", False))
        self.params.show_lower.setChecked(t.get("lower", True))
        g = d.get("diagram", {})
        if g:
            self.apply_diagram_settings(g)
        self.views = [SavedView.from_dict(v) for v in d.get("views", [])]
        self._active_view = None
        self._rebuild_views_bar()
        self.images = []
        for c in d.get("images", []):
            c = dict(c)
            c["image_path"] = res(c["image_path"])
            self.images.append(ImageEntry(Calibration.from_dict(c)))
        self._refresh_image_list()
        self.zpf = []
        zp = [res(z["path"]) for z in d.get("zpf", [])]
        if zp:
            ds, msgs = load_paths([p for p in zp if os.path.exists(p)])
            en = {res(z["path"]): z.get("enabled", True) for z in d.get("zpf", [])}
            for x in ds:
                x.enabled = en.get(x.path, True)
            self.zpf = ds
        self.zpf_tol.setValue(d.get("zpf_tolerance", 5))
        self._refresh_zpf_list()
        ed = d.get("unsaved_edits", {})
        if self.doc is not None and (ed.get("vv") or ed.get("coef")):
            for r in self.doc.rows:
                if r.kind == "VV" and r.symbol in ed.get("vv", {}):
                    self.doc.set_value(r, ed["vv"][r.symbol])
                elif r.kind == "COEF" and str(r.rid) in ed.get("coef", {}):
                    self.doc.set_value(r, ed["coef"][str(r.rid)])
            self._log("restored unsaved edits from the project")
            self.params.rebuild()
            self._update_title()
        self.replot()

    # ===================================================================================== misc
    def _log(self, msg: str):
        self.log.appendPlainText(time.strftime("%H:%M:%S  ") + msg)

    def closeEvent(self, ev):
        if not self._confirm_discard():
            ev.ignore()
            return
        self.runner.shutdown()
        super().closeEvent(ev)


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    app = QApplication(argv)
    app.setApplicationName("tdb-tweak")
    w = MainWindow()
    w.show()
    for a in argv[1:]:
        if a.endswith(".tdbtweak"):
            w.open_project(a)
        elif a.lower().endswith(".tdb"):
            w.open_tdb(a)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
