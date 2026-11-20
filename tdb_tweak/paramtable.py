"""The parameter table: VV symbol | meaning | value."""
from __future__ import annotations

from typing import List, Optional, Sequence

from PyQt6.QtCore import QAbstractTableModel, QModelIndex, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QKeySequence, QUndoCommand, QUndoStack
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QHBoxLayout, QHeaderView, QLabel,
                             QLineEdit, QPushButton, QTableView, QVBoxLayout, QWidget)

from .tdbmodel import Row, TDBDocument, fmt_number

SORT_MODES = ["VV number", "Phase → term (G, L0, L1, …, TC, BMAGN) → a, b, c…",
              "Phase → constituents → term → a, b, c…"]

LOWER_BG = QColor(255, 210, 210)
LOWER_BG_DARK = QColor(110, 40, 40)


class ParamModel(QAbstractTableModel):
    valueEdited = pyqtSignal(object, float, float)        # row, old, new (request; applied via undo stack)

    HEADERS = ["Symbol", "Meaning", "Value"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.doc: Optional[TDBDocument] = None
        self.rows: List[Row] = []
        self.elements: List[str] = []

    def set_rows(self, doc: Optional[TDBDocument], rows: List[Row], elements: Sequence[str]):
        self.beginResetModel()
        self.doc, self.rows, self.elements = doc, rows, list(elements)
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return 3

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self.HEADERS[section]
        return None

    def flags(self, index):
        f = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if index.column() == 2:
            f |= Qt.ItemFlag.ItemIsEditable
        return f

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or self.doc is None:
            return None
        r = self.rows[index.row()]
        c = index.column()
        if role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole):
            if c == 0:
                return r.symbol if r.kind == "VV" else "—"
            if c == 1:
                return r.meaning()
            if c == 2:
                return fmt_number(self.doc.value(r))
        if role == Qt.ItemDataRole.BackgroundRole and self.doc.is_lower_order(r, self.elements):
            return QBrush(LOWER_BG)
        if role == Qt.ItemDataRole.ForegroundRole:
            if self.doc.is_lower_order(r, self.elements):
                return QBrush(QColor(20, 20, 20))
            if c == 0 and r.kind != "VV":
                return QBrush(QColor(140, 140, 140))
        if role == Qt.ItemDataRole.FontRole and self.doc.is_modified(r):
            f = QFont()
            f.setBold(True)
            return f
        if role == Qt.ItemDataRole.TextAlignmentRole and c == 2:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role == Qt.ItemDataRole.ToolTipRole:
            cmd = self.doc.commands[r.command_index].text.strip()
            tip = cmd if len(cmd) < 600 else cmd[:600] + " …"
            extra = []
            if self.doc.is_modified(r):
                extra.append(f"original value: {fmt_number(self.doc.original_value(r))}")
            if self.doc.is_lower_order(r, self.elements):
                extra.append("lower-order parameter: it also changes every sub-system that contains it")
            uses = [x for x in self.doc.rows if r.kind == "VV" and x.symbol == r.symbol]
            if len(uses) > 1:
                extra.append(f"{r.symbol} is used in {len(uses)} places; editing one edits all")
            return "\n".join([tip] + extra)
        return None

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if role != Qt.ItemDataRole.EditRole or index.column() != 2 or self.doc is None:
            return False
        try:
            new = float(str(value).strip().replace("D", "E").replace("d", "e"))
        except ValueError:
            return False
        r = self.rows[index.row()]
        old = self.doc.value(r)
        if new == old:
            return False
        self.valueEdited.emit(r, old, new)
        return True

    def refresh_values(self):
        if self.rows:
            self.dataChanged.emit(self.index(0, 0), self.index(len(self.rows) - 1, 2))


class _EditCommand(QUndoCommand):
    def __init__(self, panel: "ParamPanel", row: Row, old: float, new: float):
        super().__init__(f"{row.symbol if row.kind == 'VV' else row.meaning()} = {fmt_number(new)}")
        self.panel, self.row, self.old, self.new = panel, row, old, new

    def redo(self):
        self.panel._apply(self.row, self.new)

    def undo(self):
        self.panel._apply(self.row, self.old)


class ParamPanel(QWidget):
    """Table + sort/filter controls. Emits ``changed(needs_rebuild)`` after a value changes."""
    changed = pyqtSignal(bool)
    committed = pyqtSignal()
    subsystemRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.doc: Optional[TDBDocument] = None
        self.elements: List[str] = []
        self.undo_stack = QUndoStack(self)

        self.subsystem_btn = QPushButton("Elements: —")
        self.subsystem_btn.setToolTip("Choose the elements whose parameters are edited (e.g. Fe-Ni-Cr of a "
                                      "Fe-Ni-Co-Cr database)")
        self.subsystem_btn.clicked.connect(self.subsystemRequested)
        self.sort_combo = QComboBox()
        self.sort_combo.addItems(SORT_MODES)
        self.sort_combo.setCurrentIndex(1)
        self.show_nonvv = QCheckBox("Non-VV coefficients")
        self.show_nonvv.setToolTip("Show plain numbers of PARAMETER expressions (not ESPEI VV symbols)")
        self.show_lower = QCheckBox("Lower-order parameters")
        self.show_lower.setToolTip("Show parameters of sub-systems (unaries/binaries of a ternary); "
                                   "they are highlighted red")
        self.show_lower.setChecked(True)
        self.show_nonvv.setChecked(False)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter (phase, symbol, element …)")
        self.search.setClearButtonEnabled(True)

        self.model = ParamModel(self)
        self.model.valueEdited.connect(self._on_value_edited)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked
                                   | QAbstractItemView.EditTrigger.EditKeyPressed
                                   | QAbstractItemView.EditTrigger.AnyKeyPressed)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(22)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        h.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        self.table.setColumnWidth(2, 150)
        self.table.setAlternatingRowColors(True)
        self.count_label = QLabel("")

        top = QHBoxLayout()
        top.addWidget(self.subsystem_btn)
        top.addWidget(QLabel("Sort:"))
        top.addWidget(self.sort_combo, 1)
        opts = QHBoxLayout()
        opts.addWidget(QLabel("Show:"))
        opts.addWidget(self.show_nonvv)
        opts.addWidget(self.show_lower)
        opts.addStretch(1)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addLayout(top)
        lay.addLayout(opts)
        lay.addWidget(self.search)
        lay.addWidget(self.table, 1)
        lay.addWidget(self.count_label)

        for w in (self.sort_combo,):
            w.currentIndexChanged.connect(self.rebuild)
        for w in (self.show_nonvv, self.show_lower):
            w.toggled.connect(self.rebuild)
        self.search.textChanged.connect(self.rebuild)

    # -- content ------------------------------------------------------------------------------------------
    def set_document(self, doc: Optional[TDBDocument], elements: Sequence[str]):
        self.doc = doc
        self.undo_stack.clear()
        self.set_elements(elements)
        if doc is not None and not any(r.kind == "VV" for r in doc.rows):
            self.show_nonvv.setChecked(True)   # nothing else to show

    def set_elements(self, elements: Sequence[str]):
        self.elements = sorted(e.upper() for e in elements)
        self.subsystem_btn.setText("Elements: " + ("-".join(e.capitalize() for e in self.elements) or "—"))
        self.rebuild()

    def rebuild(self):
        doc = self.doc
        if doc is None:
            self.model.set_rows(None, [], [])
            self.count_label.setText("")
            return
        rows = doc.rows_in(self.elements)
        if not self.show_nonvv.isChecked():
            rows = [r for r in rows if r.kind == "VV"]
        if not self.show_lower.isChecked():
            rows = [r for r in rows if not doc.is_lower_order(r, self.elements)]
        q = self.search.text().strip().upper()
        if q:
            rows = [r for r in rows if q in r.symbol.upper() or q in r.meaning().upper()
                    or q in r.owner_name.upper()]
        mode = self.sort_combo.currentIndex()
        if mode == 0:
            rows.sort(key=lambda r: r.vv_sort_key())
        elif mode == 1:
            rows.sort(key=lambda r: r.phase_sort_key())
        else:
            rows.sort(key=lambda r: r.phase_sort_key(by_constituents=True))
        self.model.set_rows(doc, rows, self.elements)
        self._n_rows = len(rows)
        self._n_lower = sum(1 for r in rows if doc.is_lower_order(r, self.elements))
        self.update_count()

    def update_count(self):
        if self.doc is None:
            self.count_label.setText("")
            return
        self.count_label.setText(f"{self._n_rows} rows ({self._n_lower} lower-order), "
                                 f"{self.doc.modified_count()} modified  ·  {len(self.doc.vvdefs)} VV symbols in file")

    # -- editing ------------------------------------------------------------------------------------------
    def _on_value_edited(self, row: Row, old: float, new: float):
        self.undo_stack.push(_EditCommand(self, row, old, new))
        self.committed.emit()

    def _apply(self, row: Row, value: float):
        rebuild = self.doc.set_value(row, value)
        self.model.refresh_values()
        self.update_count()
        self.changed.emit(rebuild)

    def revert_all(self):
        if self.doc is None:
            return
        self.undo_stack.beginMacro("revert all")
        for r in self.doc.rows:
            if self.doc.is_modified(r):
                if r.kind == "VV" and self.doc.vv_values[r.symbol] == self.doc.vv_orig[r.symbol]:
                    continue
                self.undo_stack.push(_EditCommand(self, r, self.doc.value(r), self.doc.original_value(r)))
        self.undo_stack.endMacro()
