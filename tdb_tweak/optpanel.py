"""The "Optimize" panel: move binary invariants / congruent points to targets by adjusting chosen parameters.

Several targets can be optimised at once.  A target that is not dragged keeps its current values as the goal,
so it holds that feature in place while the others move."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox,
                             QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QScrollArea, QSpinBox,
                             QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from . import units as U
from .optimize import (DEFAULT_FLOORS, InvariantTarget, OptIteration, OptResult, OptSettings, Variable, limit_for)
from .tdbmodel import BASIS_LETTER, Row, fmt_number

FLOOR_UNITS = {"const": "J/mol", "T": "J/mol/K", "TLNT": "J/mol/K", "T2": "J/mol/K²", "T3": "J/mol/K³",
               "TINV": "J·K/mol"}
NORM_PRESETS = ["0 — off", "0.001 — only stops blow-ups", "0.05 — mild", "1 — strong", "100 — very strong"]


def _dspin(lo, hi, val, dec, step, suffix=""):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    s.setKeyboardTracking(False)
    if suffix:
        s.setSuffix(suffix)
    return s


def diagram_tag(e: Dict) -> str:
    req = e.get("req")
    els = "-".join(x.capitalize() for x in e.get("elements", []))
    if req is not None and req.kind == "ternary":
        return f"{els} {req.temperature:.6g} K"
    return els


def entry_label(e: Dict) -> str:
    if e["kind"] == "boundary":
        return f"{e['names'][0]} boundary of {e['names'][0]} + {e['names'][1]}"
    if e["kind"] == "congruent":
        return " = ".join(e["names"]) + "  (congruent)"
    if e["kind"] == "tietriangle":
        return " + ".join(e["names"]) + "  (tie triangle)"
    return " + ".join(e["names"])


def base_phase(name: str) -> str:
    return name.split("#")[0]


def fixed_phases(e: Dict) -> List[str]:
    """Phases whose composition cannot move (line compounds)."""
    sp = e.get("spaces") or {}
    return [n for n in e["names"] if sp.get(base_phase(n), {}).get("rank", 2) == 0]


def new_entry(item: Dict) -> Dict:
    """A target from a point of the map; its wanted values start at the current ones (= hold in place).

    item: kind, names, T, cur_x (keys: phase for binaries, "PHASE|EL" for ternary corners), spaces, ..."""
    e = dict(item)
    e["want_T"] = float(item["T"])
    fixed = set(fixed_phases(e))
    if item["kind"] == "congruent":
        keys = [] if item.get("fixed_x") else [item["names"][0]]
    elif item["kind"] == "boundary" and item.get("direction") is not None:      # ternary boundary node
        els = item["elements"]
        keys = [f"node|{els[1]}", f"node|{els[2]}"]
        e["cur_x"] = dict(item["cur_x"])
        e["cur_x"][f"node|{els[1]}"] = item["cur_x"][f"{item['names'][0]}|{els[1]}"]
        e["cur_x"][f"node|{els[2]}"] = item["cur_x"][f"{item['names'][0]}|{els[2]}"]
    elif item["kind"] == "boundary":
        keys = [item["names"][0]]
    else:
        keys = [k for k in item["cur_x"] if k.split("|")[0] not in fixed]
    e["want_x"] = {k: float(e["cur_x"][k] if k in e.get("cur_x", {}) else item["cur_x"][k]) for k in keys}
    e["use_T"] = item["kind"] != "tietriangle"
    e["use_x"] = {k: True for k in keys}
    return e


def key_label(e: Dict, k: str) -> str:
    if k.startswith("node|"):
        return f"node x {k.split('|')[1].capitalize()}"
    if e["kind"] == "boundary":
        return f"x({k}) at that T"
    if e["kind"] == "congruent":
        return f"x ({' = '.join(e['names'])})"
    if "|" in k:
        ph, el = k.split("|")
        return f"x {el.capitalize()} ({ph})"
    return f"x({k})"


class OptPanel(QWidget):
    runRequested = pyqtSignal()
    stopRequested = pyqtSignal()
    applyIteration = pyqtSignal(int)
    addRequested = pyqtSignal(int)          # index into the "points on this map" list
    removeRequested = pyqtSignal(int)       # index into entries
    clearRequested = pyqtSignal()
    targetsEdited = pyqtSignal()
    useSelectionRequested = pyqtSignal()
    useVisibleRequested = pyqtSignal()
    phaseLocksChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.t_unit = "K"
        self.entries: List[Dict] = []
        self.result: Optional[OptResult] = None
        self._rows: List[Row] = []
        self._phases_of: Dict[str, set] = {}             # symbol -> phases whose parameters use it
        self.doc = None                                   # the TDBDocument (its locks are the frozen state)
        self._phase_cbs: Dict[str, QCheckBox] = {}
        self._row_meta: List[Tuple[int, str]] = []     # (entry index, "head" / "T" / phase)
        self._building = False

        inner = QWidget()
        lay = QVBoxLayout(inner)

        # --- targets ---
        g1 = QGroupBox("Targets (binary invariants and congruent points, ternary tie triangles)")
        l1 = QVBoxLayout(g1)
        hint = QLabel("Right-click an invariant line or a congruent point (◇) of a binary, or inside a tie triangle of "
                      "a ternary section → <i>Optimize…</i> adds it here. Drag its handles to where it should be. "
                      "A target you leave where it is holds that feature in place while the others move. Grey "
                      "handles are fixed by stoichiometry; handles of 1-D stoichiometric phases slide along their line.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#666;")
        add_row = QHBoxLayout()
        self.points_combo = QComboBox()
        self.points_combo.setToolTip("Invariants and congruent points of the current binary map")
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(lambda: self.addRequested.emit(self.points_combo.currentIndex()))
        add_row.addWidget(QLabel("On this map:"))
        add_row.addWidget(self.points_combo, 1)
        add_row.addWidget(add_btn)
        self.tgt_table = QTableWidget(0, 4)
        self.tgt_table.setHorizontalHeaderLabels(["Use", "Quantity", "Current", "Want"])
        self.tgt_table.verticalHeader().setVisible(False)
        self.tgt_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.tgt_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.tgt_table.setMinimumHeight(280)
        self.tgt_table.itemChanged.connect(self._on_item_changed)
        rm_row = QHBoxLayout()
        rm = QPushButton("Remove selected target")
        clr = QPushButton("Clear all")
        rm.clicked.connect(self._remove_selected)
        clr.clicked.connect(self.clearRequested)
        rm_row.addWidget(rm)
        rm_row.addWidget(clr)
        self.export_btn = QPushButton("Export as ESPEI data…")
        self.export_btn.setToolTip("Write the targets (at their wanted positions) as ESPEI ZPF datasets, one per system "
                                   "(first version)")
        rm_row.addWidget(self.export_btn)
        tol = QHBoxLayout()
        self.t_tol = _dspin(0.001, 500, 1.0, 3, 0.5, " K")
        self.x_tol = _dspin(0.0001, 10, 0.1, 4, 0.05, " at.%")
        tol.addWidget(QLabel("Tolerance:"))
        tol.addWidget(self.t_tol)
        tol.addWidget(self.x_tol)
        l1.addWidget(hint)
        l1.addLayout(add_row)
        l1.addWidget(self.tgt_table)
        l1.addLayout(rm_row)
        l1.addLayout(tol)
        lay.addWidget(g1)

        # --- variables ---
        g2 = QGroupBox("Parameters to vary")
        l2 = QVBoxLayout(g2)
        b = QHBoxLayout()
        vis = QPushButton("VV rows shown in table")
        vis.setToolTip("Every VV symbol currently listed in the parameter table (after its filters)")
        sel = QPushButton("Selected table rows")
        sel.setToolTip("Only the rows selected in the parameter table (VV or plain coefficients)")
        vis.clicked.connect(self.useVisibleRequested)
        sel.clicked.connect(self.useSelectionRequested)
        b.addWidget(vis)
        b.addWidget(sel)
        self.var_table = QTableWidget(0, 4)
        self.var_table.setHorizontalHeaderLabels(["Use", "Parameter", "Value", "Max Δ / iter"])
        self.var_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.var_table.verticalHeader().setVisible(False)
        self.var_table.setMinimumHeight(140)
        self.var_table.setToolTip("Max Δ / iter: leave blank for the default limit below, or type an absolute value")
        l2.addLayout(b)
        l2.addWidget(self.var_table)
        self.phase_box = QWidget()
        self.phase_grid = QGridLayout(self.phase_box)
        self.phase_grid.setContentsMargins(0, 0, 0, 0)
        l2.addWidget(self.phase_box)
        lim = QHBoxLayout()
        self.rel = _dspin(0, 100000, 100, 1, 10, " %")
        lim.addWidget(QLabel("Default max change per iteration:"))
        lim.addWidget(self.rel)
        lim.addWidget(QLabel("of the value, but at least:"))
        l2.addLayout(lim)
        self.floor_edits: Dict[str, QLineEdit] = {}
        floors_row = QGridLayout()
        for k, basis in enumerate(["const", "T", "TLNT", "T2", "T3", "TINV"]):
            e = QLineEdit(fmt_number(DEFAULT_FLOORS[basis]))
            e.setToolTip(f"coefficient {BASIS_LETTER[basis]} ({FLOOR_UNITS[basis]}); "
                         f"the default corresponds to about 50 kJ/mol at 1000 K")
            self.floor_edits[basis] = e
            floors_row.addWidget(QLabel(BASIS_LETTER[basis]), k // 3, 2 * (k % 3))
            floors_row.addWidget(e, k // 3, 2 * (k % 3) + 1)
        l2.addLayout(floors_row)

        nrow = QHBoxLayout()
        self.norm = QComboBox()
        self.norm.setEditable(True)
        self.norm.addItems(NORM_PRESETS)
        self.norm.setToolTip("Adds weight × Σ ((p − centre) / floor)² to the cost; a target off by one tolerance "
                             "costs 1.\n0.001: almost no effect unless a parameter grows to ~100× its floor.\n"
                             "1: a floor-sized parameter weighs as much as one tolerance.\n"
                             "100: keeps parameters as small as possible, even at the cost of the targets.\n"
                             "With the norm on, iterations continue after the targets are met while the total "
                             "still decreases.")
        self.norm_center = QComboBox()
        self.norm_center.addItem("toward zero", "zero")
        self.norm_center.addItem("toward the starting values", "start")
        nrow.addWidget(QLabel("Norm weight"))
        nrow.addWidget(self.norm, 1)
        nrow.addWidget(self.norm_center)
        l2.addLayout(nrow)
        it = QHBoxLayout()
        self.max_iter = QSpinBox()
        self.max_iter.setRange(1, 500)
        self.max_iter.setValue(10)
        self.reject = QCheckBox("Reject steps that make a target metastable")
        self.reject.setChecked(True)
        self.reject.setToolTip("Checked: such steps are shortened instead (the run stops if none is possible).\n"
                               "Unchecked: they are accepted and flagged in the history.")
        it.addWidget(QLabel("Max iterations"))
        it.addWidget(self.max_iter)
        it.addWidget(self.reject)
        l2.addLayout(it)
        hr = QHBoxLayout()
        self.hold_others = QCheckBox("Hold everything else")
        self.hold_others.setToolTip("Every other invariant, congruent point and tie triangle of the targets' diagrams "
                                    "is held where it is (within 5 K / 1 at.%) while the targets move")
        self.max_total = _dspin(0.1, 1000, 5, 1, 1, " × limit")
        self.max_total.setToolTip("Over one run, a parameter moves at most this many per-iteration limits from where "
                                  "it started (stops one parameter from running away when the others barely act)")
        hr.addWidget(self.hold_others)
        hr.addStretch(1)
        dr = QHBoxLayout()
        self.fit_zpf = QCheckBox("Fit the loaded ZPF data")
        self.fit_zpf.setToolTip("Also minimise the driving forces of the loaded (ticked) ZPF datasets of the targets' "
                                "systems, σ = 1000 J/mol per point. Works without any dragged target too.")
        self.reject_art = QCheckBox("Reject steps that create artifacts")
        self.reject_art.setChecked(True)
        self.reject_art.setToolTip("A step that creates a new artifact (a solid stable at 4000-6000 K, a high-T "
                                   "miscibility gap, ...) is shortened instead; checked with Diagram ▸ Check for "
                                   "Artifacts' scan")
        dr.addWidget(self.fit_zpf)
        dr.addWidget(self.reject_art)
        dr.addStretch(1)
        self.live = QCheckBox("Live drag: parameters follow the handle")
        self.live.setChecked(True)
        self.live.setToolTip("Pressing a handle computes how every target responds to every chosen parameter (about a "
                             "second); while dragging, the parameters follow from that linear model in microseconds "
                             "and the filled dots show where the feature is predicted to be. Within about one "
                             "per-iteration limit of change it is accurate; beyond, the dots stop following: release "
                             "and press again. One undo step per drag.")
        dr2 = QHBoxLayout()
        dr2.addWidget(self.live)
        dr2.addStretch(1)
        hr.addWidget(QLabel("Total change ≤"))
        hr.addWidget(self.max_total)
        l2.addLayout(hr)
        l2.addLayout(dr)
        l2.addLayout(dr2)
        lay.addWidget(g2)

        # --- run + history ---
        rb = QHBoxLayout()
        self.run_btn = QPushButton("Optimize")
        self.run_btn.setStyleSheet("font-weight:bold;")
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.runRequested)
        self.stop_btn.clicked.connect(self.stopRequested)
        rb.addWidget(self.run_btn)
        rb.addWidget(self.stop_btn)
        lay.addLayout(rb)
        self.status = QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        g3 = QGroupBox("Iterations (select one to apply it instead of the last)")
        l3 = QVBoxLayout(g3)
        self.hist = QTableWidget(0, 4)
        self.hist.verticalHeader().setVisible(False)
        self.hist.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.hist.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.hist.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.hist.setMinimumHeight(150)
        self.hist.itemSelectionChanged.connect(self._show_changes)
        ab = QHBoxLayout()
        self.apply_btn = QPushButton("Apply selected iteration")
        self.revert_btn = QPushButton("Revert to start")
        self.apply_btn.clicked.connect(lambda: self.applyIteration.emit(self.selected_iteration()))
        self.revert_btn.clicked.connect(lambda: self.applyIteration.emit(0))
        ab.addWidget(self.apply_btn)
        ab.addWidget(self.revert_btn)
        self.changes = QTableWidget(0, 4)
        self.changes.setHorizontalHeaderLabels(["Parameter", "Start", "Value", "Δ / limit"])
        self.changes.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.changes.verticalHeader().setVisible(False)
        self.changes.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.changes.setMinimumHeight(140)
        l3.addWidget(self.hist)
        l3.addLayout(ab)
        l3.addWidget(QLabel("Parameter changes at the selected iteration (largest first):"))
        l3.addWidget(self.changes)
        lay.addWidget(g3)
        lay.addStretch(1)

        sc = QScrollArea()
        sc.setWidgetResizable(True)
        sc.setWidget(inner)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(sc)

    # ----------------------------------------------------------------------------------------- targets
    def set_points(self, items: List[Dict], t_unit: str):
        self.points_combo.clear()
        for it in items:
            self.points_combo.addItem(f"{entry_label(it)}   {U.t_from_K(it['T'], t_unit):.6g} {t_unit}")

    def set_entries(self, entries: List[Dict], t_unit: str):
        """Rebuild the target table (entries are shared with the controller and edited in place)."""
        self.entries = entries
        self.t_unit = t_unit
        self._building = True
        t = self.tgt_table
        t.clearSpans()
        meta = []
        for i, e in enumerate(entries):
            meta.append((i, "head"))
            if e["kind"] != "tietriangle" and len(e.get("elements", [])) != 3:
                meta.append((i, "T"))
            for k in e["want_x"]:
                meta.append((i, k))
        t.setRowCount(len(meta))
        bold = QFont()
        bold.setBold(True)
        for r, (i, f) in enumerate(meta):
            e = entries[i]
            if f == "head":
                txt = f"{i + 1}.  {entry_label(e)}   [{diagram_tag(e)}]"
                if e["kind"] == "congruent" and e.get("fixed_x"):
                    txt += f"   x fixed by {e['compound']}"
                elif fixed_phases(e):
                    txt += f"   fixed composition: {', '.join(fixed_phases(e))}"
                it = QTableWidgetItem(txt)
                it.setFont(bold)
                it.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                it.setBackground(QBrush(QColor(245, 230, 242)))
                t.setItem(r, 0, it)
                t.setSpan(r, 0, 1, 4)
                continue
            use = QTableWidgetItem()
            use.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            on = e["use_T"] if f == "T" else e["use_x"][f]
            use.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            name = QTableWidgetItem((f"at T ({t_unit})" if e["kind"] == "boundary" else f"T ({t_unit})")
                                    if f == "T" else key_label(e, f))
            name.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            cur = QTableWidgetItem("")
            cur.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            want = QTableWidgetItem("")
            for c, w in enumerate((use, name, cur, want)):
                t.setItem(r, c, w)
        self._row_meta = meta
        self._building = False
        self.refresh_values()

    def refresh_values(self):
        """Update the Current and Want columns from the entries (after a drag or a new map)."""
        self._building = True
        for r, (i, f) in enumerate(self._row_meta):
            if f == "head" or i >= len(self.entries):
                continue
            e = self.entries[i]
            if f == "T":
                cur = f"{U.t_from_K(e['T'], self.t_unit):.6g}"
                want = f"{U.t_from_K(e['want_T'], self.t_unit):.6g}"
            else:
                cur = f"{e['cur_x'][f] * 100:.4g} %"
                want = f"{e['want_x'][f] * 100:.4g}"
            self.tgt_table.item(r, 2).setText(cur)
            self.tgt_table.item(r, 3).setText(want)
        self._building = False

    def _on_item_changed(self, item: QTableWidgetItem):
        if self._building or item.row() >= len(self._row_meta):
            return
        i, f = self._row_meta[item.row()]
        if f == "head":
            return
        e = self.entries[i]
        if item.column() == 0:
            on = item.checkState() == Qt.CheckState.Checked
            if f == "T":
                e["use_T"] = on
            else:
                e["use_x"][f] = on
        elif item.column() == 3:
            m = re.match(r"\s*([-+0-9.eE]+)", item.text())
            if m:
                val = float(m.group(1))
                if f == "T":
                    e["want_T"] = float(U.t_to_K(val, self.t_unit))
                else:
                    e["want_x"][f] = float(np.clip(val / 100, 0, 1))
            self.refresh_values()
        self.targetsEdited.emit()

    def _remove_selected(self):
        rows = self.tgt_table.selectionModel().selectedRows()
        if rows and rows[0].row() < len(self._row_meta):
            self.removeRequested.emit(self._row_meta[rows[0].row()][0])

    def build_targets(self) -> List[InvariantTarget]:
        out = []
        for e in self.entries:
            cx = e["cur_x"]
            corners = {}
            if e["kind"] == "tietriangle":
                els = e["elements"]
                corners = {n: (cx[f"{n}|{els[1]}"], cx[f"{n}|{els[2]}"]) for n in e["names"]}
            elif e["kind"] in ("invariant", "boundary") and e.get("direction") is None:
                corners = {n: (cx[n],) for n in e["names"]}
            point = direction = None
            x_target = {k: (v if e["use_x"][k] else None) for k, v in e["want_x"].items()}
            if e["kind"] == "boundary" and e.get("direction") is not None:              # ternary boundary node
                els = e["elements"]
                a, b = e["names"]
                point = (e["want_x"][f"node|{els[1]}"], e["want_x"][f"node|{els[2]}"])
                direction = tuple(e["direction"])
                use = e["use_x"].get(f"node|{els[1]}", True) or e["use_x"].get(f"node|{els[2]}", True)
                x_target = {f"{a}|across": float(np.dot(point, direction)) if use else None}
                corners = {a: (cx[f"{a}|{els[1]}"], cx[f"{a}|{els[2]}"]), b: (cx[f"{b}|{els[1]}"], cx[f"{b}|{els[2]}"])}
                T_target = None
            elif e["kind"] == "boundary":
                T_target = e["want_T"]                      # where the boundary is evaluated (a condition)
            elif e["kind"] == "tietriangle":
                T_target = None
            else:
                T_target = e["want_T"] if e["use_T"] else None
            out.append(InvariantTarget(
                tuple(e["names"]), float(e["T"]), T_target=T_target,
                x_target=x_target,
                T_tol=self.t_tol.value(), x_tol=self.x_tol.value() / 100, kind=e["kind"],
                x_guess=float(next(iter(cx.values()))) if cx else 0.5, corners_guess=corners, req=e.get("req"),
                point=point, direction=direction))
        return out

    # ----------------------------------------------------------------------------------------- variables
    def set_variable_rows(self, rows: List[Row], values: Dict[str, float],
                          phases_of: Optional[Dict[str, set]] = None, doc=None):
        self._rows = rows
        if doc is not None:
            self.doc = doc
        self._phases_of = phases_of or {r.symbol: {r.phase or "(functions)"} for r in rows}
        self.var_table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            use = QTableWidgetItem()
            use.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            use.setCheckState(Qt.CheckState.Checked)
            name = QTableWidgetItem(f"{r.symbol if r.kind == 'VV' else '—'}  {r.meaning()}")
            name.setFlags(Qt.ItemFlag.ItemIsEnabled)
            val = QTableWidgetItem(fmt_number(values.get(r.symbol, 0.0)))
            val.setFlags(Qt.ItemFlag.ItemIsEnabled)
            lim = QTableWidgetItem("")
            for j, it in enumerate((use, name, val, lim)):
                self.var_table.setItem(i, j, it)
        self._build_phase_boxes()
        self._apply_frozen()

    # phase freezing = locking every parameter of a phase (shared with the parameter table) -------------
    def _build_phase_boxes(self):
        while self.phase_grid.count():
            w = self.phase_grid.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        self._phase_cbs = {}
        phases = sorted(set().union(*self._phases_of.values())) if self._phases_of else []
        if not phases:
            return
        lab = QLabel("Vary parameters of:")
        lab.setToolTip("Unticking a phase locks every parameter of that phase (also in the parameter table); a "
                       "symbol shared with a locked phase is locked too. Partly ticked: some of its parameters "
                       "are locked.")
        self.phase_grid.addWidget(lab, 0, 0, 1, 3)
        for k, ph in enumerate(phases):
            cb = QCheckBox(ph)
            cb.setTristate(True)
            cb.clicked.connect(lambda _=False, ph=ph: self._toggle_phase(ph))
            self.phase_grid.addWidget(cb, 1 + k // 3, k % 3)
            self._phase_cbs[ph] = cb
        self.refresh_locks()

    def _phase_rows(self, phase: str):
        return self.doc.rows_of_phase(phase) if self.doc is not None else []

    def _toggle_phase(self, phase: str):
        rows = self._phase_rows(phase)
        if not rows:
            return
        all_locked = all(self.doc.is_locked(r) for r in rows)
        self.doc.set_locked(rows, not all_locked)
        self.refresh_locks()
        self.phaseLocksChanged.emit()

    def refresh_locks(self):
        """Phase boxes and greyed rows from the document's locks."""
        for ph, cb in self._phase_cbs.items():
            rows = self._phase_rows(ph)
            n = sum(1 for r in rows if self.doc.is_locked(r)) if rows else 0
            st = Qt.CheckState.Checked if n == 0 else (Qt.CheckState.Unchecked if n == len(rows)
                                                         else Qt.CheckState.PartiallyChecked)
            cb.blockSignals(True)
            cb.setCheckState(st)
            cb.blockSignals(False)
        self._apply_frozen()

    def frozen_by(self, r: Row) -> set:
        return {"locked"} if (self.doc is not None and self.doc.is_locked(r)) else set()

    def _apply_frozen(self):
        grey = QBrush(QColor(160, 160, 160))
        normal = QBrush(self.palette().color(self.foregroundRole()))
        for i, r in enumerate(self._rows):
            fz = bool(self.frozen_by(r))
            for j in range(4):
                it = self.var_table.item(i, j)
                if it is None:
                    continue
                it.setForeground(grey if fz else normal)
                it.setToolTip("locked (unlock it in the parameter table or tick its phase)" if fz else "")
            use = self.var_table.item(i, 0)
            if use is not None:
                use.setFlags(Qt.ItemFlag.ItemIsUserCheckable |
                             (Qt.ItemFlag.NoItemFlags if fz else Qt.ItemFlag.ItemIsEnabled))

    def update_variable_values(self, values: Dict[str, float]):
        for i, r in enumerate(self._rows):
            it = self.var_table.item(i, 2)
            if it is not None:
                it.setText(fmt_number(values.get(r.symbol, 0.0)))

    def chosen(self) -> List[Tuple[Row, Optional[float]]]:
        out = []
        for i, r in enumerate(self._rows):
            if self.var_table.item(i, 0).checkState() != Qt.CheckState.Checked or self.frozen_by(r):
                continue
            txt = self.var_table.item(i, 3).text().strip()
            try:
                lim = float(txt) if txt else None
            except ValueError:
                lim = None
            out.append((r, lim))
        return out

    def norm_weight(self) -> float:
        m = re.match(r"\s*([0-9.eE+-]+)", self.norm.currentText())
        try:
            return max(0.0, float(m.group(1))) if m else 0.0
        except ValueError:
            return 0.0

    def settings(self) -> OptSettings:
        floors = {}
        for b, e in self.floor_edits.items():
            try:
                floors[b] = abs(float(e.text()))
            except ValueError:
                floors[b] = DEFAULT_FLOORS[b]
        return OptSettings(max_iter=self.max_iter.value(), rel_limit=self.rel.value() / 100, floors=floors,
                           check_metastable=True, reject_metastable=self.reject.isChecked(),
                           norm_weight=self.norm_weight(), norm_center=self.norm_center.currentData(),
                           max_total=self.max_total.value(), hold_others=self.hold_others.isChecked(),
                           fit_zpf=self.fit_zpf.isChecked(), reject_artifacts=self.reject_art.isChecked())

    # ----------------------------------------------------------------------------------------- history
    def clear_history(self):
        self.result = None
        self.hist.setRowCount(0)
        self.changes.setRowCount(0)

    def _hist_headers(self, n_targets: int):
        heads = ["#", "max |r|", "norm / ZPF"] + [f"{k + 1}: T / x" for k in range(n_targets)] + ["note"]
        self.hist.setColumnCount(len(heads))
        self.hist.setHorizontalHeaderLabels(heads)
        self.hist.horizontalHeader().setSectionResizeMode(len(heads) - 1, QHeaderView.ResizeMode.Stretch)

    def add_iteration(self, it: OptIteration):
        if self.hist.rowCount() == 0:
            self._hist_headers(len(it.points))
        n = self.hist.rowCount()
        self.hist.insertRow(n)
        cells = [str(it.index), f"{it.residual:.3g}",
                 f"{it.norm:.3g}" + (f" / {it.zpf_rms:.0f} J/mol" if it.zpf_rms is not None else "")]
        for T, x in it.points:
            xs = sorted(set(round(v * 100, 4) for v in x.values()))
            cells.append(f"{U.t_from_K(T, self.t_unit):.6g} / " + ", ".join(f"{v:.4g}" for v in xs))
        cells.append(it.note)
        for j, c in enumerate(cells):
            item = QTableWidgetItem(c)
            if it.residual <= 1.0:
                item.setForeground(QBrush(QColor(25, 135, 84)))
            if "metastable" in it.note:
                item.setBackground(QBrush(QColor(255, 225, 190)))
            self.hist.setItem(n, j, item)

    def set_result(self, res: OptResult, applied: int):
        self.result = res
        self.hist.setRowCount(0)
        for it in res.iterations:
            self.add_iteration(it)
        self.hist.selectRow(applied)

    def selected_iteration(self) -> int:
        rows = self.hist.selectionModel().selectedRows()
        return rows[0].row() if rows else max(0, self.hist.rowCount() - 1)

    def _show_changes(self):
        res = self.result
        self.changes.setRowCount(0)
        if res is None:
            return
        k = self.selected_iteration()
        if k >= len(res.iterations):
            return
        start, it = res.iterations[0], res.iterations[k]
        s = self.settings()
        rows = []
        for var in res.variables:
            a, b = start.values[var.name], it.values[var.name]
            lim = limit_for(var, a, s)
            rows.append((abs(b - a) / lim if lim else 0.0, var, a, b))
        rows.sort(key=lambda t: -t[0])
        self.changes.setRowCount(len(rows))
        for i, (ratio, var, a, b) in enumerate(rows):
            cells = [var.label or var.name, fmt_number(round(a, 10)), fmt_number(round(b, 10)), f"{ratio:.2f}"]
            for j, c in enumerate(cells):
                item = QTableWidgetItem(c)
                if ratio > 1.0:
                    item.setBackground(QBrush(QColor(255, 210, 210)))
                self.changes.setItem(i, j, item)

    def set_running(self, on: bool, text: str = ""):
        self.run_btn.setEnabled(not on)
        self.stop_btn.setEnabled(on)
        if text:
            self.status.setText(text)
