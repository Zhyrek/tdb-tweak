"""Dialog to calibrate a literature phase-diagram image (axes, units, bounds, reference points)."""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap, QTransform
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
                             QGraphicsEllipseItem, QGraphicsPathItem, QGraphicsPixmapItem, QGraphicsScene,
                             QGraphicsSimpleTextItem, QGraphicsView, QGridLayout, QGroupBox, QHBoxLayout,
                             QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton, QSplitter,
                             QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from . import units as U
from .calibration import CalPoint, Calibration, apply, detect_frame, place_from_frame

KINDS = [("binary", "Binary T–x"), ("ternary", "Ternary isothermal section"), ("isopleth", "Isopleth (T–x)")]


def load_rgba(path: str) -> np.ndarray:
    img = QImage(path)
    if img.isNull():
        raise ValueError(f"cannot read image {path}")
    img = img.convertToFormat(QImage.Format.Format_RGBA8888)
    w, h = img.width(), img.height()
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes())
    arr = np.frombuffer(ptr, np.uint8).reshape(h, img.bytesPerLine())[:, :w * 4].reshape(h, w, 4)
    return arr.copy()


def _f(text: str) -> Optional[float]:
    try:
        return float(text.strip())
    except (ValueError, AttributeError):
        return None


class ImageView(QGraphicsView):
    clicked = pyqtSignal(float, float)
    dragged = pyqtSignal(int, float, float)        # marker index, new scene position

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._pix: Optional[QGraphicsPixmapItem] = None
        self._markers: List = []
        self._overlay: Optional[QGraphicsPathItem] = None
        self._pan = None
        self._press = None
        self._marker_pos: List[tuple] = []          # (index, x, y) of placed markers
        self._dragging: Optional[int] = None

    def set_image(self, path: str):
        self.scene().clear()
        self._markers, self._overlay = [], None
        self._pix = self.scene().addPixmap(QPixmap(path))
        self.setSceneRect(self._pix.boundingRect())
        self.fitInView(self._pix, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, ev):
        f = 1.25 if ev.angleDelta().y() > 0 else 0.8
        self.scale(f, f)

    def mousePressEvent(self, ev):
        if ev.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton) or \
                (ev.button() == Qt.MouseButton.LeftButton and ev.modifiers() & Qt.KeyboardModifier.ShiftModifier):
            self._pan = ev.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if ev.button() == Qt.MouseButton.LeftButton:
            for i, x, y in self._marker_pos:
                p = self.mapFromScene(QPointF(x, y))
                if (QPointF(p) - ev.position()).manhattanLength() <= 12:
                    self._dragging = i
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                    return
            self._press = ev.position()
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self._dragging is not None:
            p = self.mapToScene(ev.position().toPoint())
            self.dragged.emit(self._dragging, p.x(), p.y())
            return
        if self._pan is not None:
            d = ev.position() - self._pan
            self._pan = ev.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self._dragging is not None:
            self._dragging = None
            self.unsetCursor()
            return
        if self._pan is not None:
            self._pan = None
            self.unsetCursor()
            return
        if ev.button() == Qt.MouseButton.LeftButton and self._press is not None:
            if (ev.position() - self._press).manhattanLength() < 4:
                p = self.mapToScene(ev.position().toPoint())
                self.clicked.emit(p.x(), p.y())
            self._press = None
        super().mouseReleaseEvent(ev)

    def set_markers(self, pts: Sequence[tuple]):
        """pts: (x, y, label, selected[, index])."""
        for m in self._markers:
            self.scene().removeItem(m)
        self._markers = []
        self._marker_pos = [(p[4], p[0], p[1]) for p in pts if len(p) > 4]
        for x, y, label, sel, *_ in pts:
            col = QColor(230, 30, 30) if sel else QColor(20, 90, 230)
            pen = QPen(col, 2)
            pen.setCosmetic(True)
            path = QPainterPath()
            r = 9
            path.moveTo(-r, 0); path.lineTo(r, 0); path.moveTo(0, -r); path.lineTo(0, r)
            path.addEllipse(QPointF(0, 0), r * 0.6, r * 0.6)
            item = QGraphicsPathItem(path)
            item.setPen(pen)
            item.setPos(x, y)
            item.setFlag(item.GraphicsItemFlag.ItemIgnoresTransformations)
            item.setZValue(10)
            txt = QGraphicsSimpleTextItem(label)
            txt.setBrush(QBrush(col))
            txt.setPos(x, y)
            txt.setTransform(QTransform.fromTranslate(12, -18))
            txt.setFlag(txt.GraphicsItemFlag.ItemIgnoresTransformations)
            txt.setZValue(10)
            self.scene().addItem(item)
            self.scene().addItem(txt)
            self._markers += [item, txt]

    def set_overlay(self, polylines: Sequence[np.ndarray]):
        if self._overlay is not None:
            self.scene().removeItem(self._overlay)
            self._overlay = None
        if not polylines:
            return
        path = QPainterPath()
        for pl in polylines:
            ok = np.all(np.isfinite(pl), axis=1)
            pl = pl[ok]
            if len(pl) < 2:
                continue
            path.moveTo(QPointF(*pl[0]))
            for p in pl[1:]:
                path.lineTo(QPointF(*p))
        self._overlay = QGraphicsPathItem(path)
        pen = QPen(QColor(255, 0, 160, 200), 1.2)
        pen.setCosmetic(True)
        self._overlay.setPen(pen)
        self._overlay.setZValue(5)
        self.scene().addItem(self._overlay)


class CalibrationDialog(QDialog):
    def __init__(self, image_path: str, elements: Sequence[str], cal: Optional[Calibration] = None,
                 kind: str = "binary", diagram_elements: Optional[Sequence[str]] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Calibrate {os.path.basename(image_path)}")
        self.resize(1300, 820)
        self.elements = [e.upper() for e in elements]
        if cal is None:
            cal = Calibration(image_path, kind=kind)
            cal.elements = list(diagram_elements or self.elements[:3 if kind != "binary" else 2])
            if kind == "ternary":
                cal.mode, cal.bounds, cal.x_unit = "points", [0.0, 100.0, 0.0, 100.0], "at.%"
            cal.default_points()
        self.cal = cal
        self._selected = 0
        self._building = False

        self.view = ImageView()
        self.view.set_image(image_path)
        self.view.clicked.connect(self._on_click)
        self.view.dragged.connect(self._on_drag)

        # --- form ---
        self.kind = QComboBox()
        for k, label in KINDS:
            self.kind.addItem(label, k)
        self.el = [QComboBox() for _ in range(3)]
        for c in self.el:
            c.addItems(self.elements)
        self.t_unit = QComboBox()
        self.t_unit.addItems(U.T_UNITS)
        self.x_unit = QComboBox()
        self.x_unit.addItems(U.X_UNITS)
        self.b = [QLineEdit() for _ in range(4)]
        self.b_labels = [QLabel(), QLabel()]
        self.mode = QComboBox()
        self.mode.addItem("Two points on each axis", "axes")
        self.mode.addItem("Points with both values known", "points")
        self.persp = QCheckBox("Perspective correction (4+ points)")
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Point", "Pixel", "x", "T"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.add_pt = QPushButton("Add point")
        self.del_pt = QPushButton("Remove point")
        self.from_bounds = QPushButton("Values from bounds")
        self.detect_btn = QPushButton("Detect frame")
        self.detect_btn.setToolTip("Find the plot frame and put the axis points on its corners, with the axis bounds "
                                   "as their values (binary / isopleth); adjust what is off")
        self.info = QLabel()
        self.info.setWordWrap(True)

        form = QFormLayout()
        form.addRow("Diagram", self.kind)
        elrow = QHBoxLayout()
        self.el_labels = [QLabel("A"), QLabel("B"), QLabel("C")]
        for lab, c in zip(self.el_labels, self.el):
            elrow.addWidget(lab)
            elrow.addWidget(c)
        form.addRow("Elements", elrow)
        form.addRow("Temperature unit", self.t_unit)
        form.addRow("Composition unit", self.x_unit)
        g = QGridLayout()
        g.addWidget(self.b_labels[0], 0, 0)
        g.addWidget(self.b[0], 0, 1)
        g.addWidget(QLabel("to"), 0, 2)
        g.addWidget(self.b[1], 0, 3)
        g.addWidget(self.b_labels[1], 1, 0)
        g.addWidget(self.b[2], 1, 1)
        g.addWidget(QLabel("to"), 1, 2)
        g.addWidget(self.b[3], 1, 3)
        bounds_box = QGroupBox("Axis bounds shown (image units)")
        bounds_box.setLayout(g)
        form.addRow(bounds_box)
        form.addRow("Calibration", self.mode)
        form.addRow("", self.persp)

        help_lbl = QLabel("Select a point in the table, then click its position on the image "
                          "(wheel: zoom, right-drag or shift-drag: pan). Points need not be the frame corners: "
                          "any tick with a known value works. The magenta grid shows the bounds mapped back onto "
                          "the image; it should sit on the image's own axes and gridlines.")
        help_lbl.setWordWrap(True)
        help_lbl.setStyleSheet("color: #555;")
        btns = QHBoxLayout()
        btns.addWidget(self.add_pt)
        btns.addWidget(self.del_pt)
        btns.addWidget(self.from_bounds)
        btns.addWidget(self.detect_btn)
        self.bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.bb.accepted.connect(self._accept)
        self.bb.rejected.connect(self.reject)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addLayout(form)
        rl.addWidget(help_lbl)
        rl.addWidget(self.table, 1)
        rl.addLayout(btns)
        rl.addWidget(self.info)
        rl.addWidget(self.bb)
        split = QSplitter()
        split.addWidget(self.view)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        lay = QVBoxLayout(self)
        lay.addWidget(split)

        self._load_into_form()
        self.kind.currentIndexChanged.connect(self._on_kind)
        self.mode.currentIndexChanged.connect(self._on_mode)
        for w in (self.t_unit, self.x_unit, *self.el):
            w.currentIndexChanged.connect(self._on_form)
        for w in self.b:
            w.editingFinished.connect(self._on_form)
        self.persp.toggled.connect(self._on_form)
        self.table.itemChanged.connect(self._on_table_edit)
        self.table.itemSelectionChanged.connect(self._on_select)
        self.add_pt.clicked.connect(self._add_point)
        self.del_pt.clicked.connect(self._del_point)
        self.from_bounds.clicked.connect(self._values_from_bounds)
        self.detect_btn.clicked.connect(lambda: self._detect(verbose=True))
        self._refresh()
        if self.cal.kind != "ternary" and not any(p.placed for p in self.cal.points):
            self._detect(verbose=False)

    def _detect(self, verbose: bool):
        self._read_form()
        if self.cal.kind == "ternary":
            self.info.setText("Frame detection works for binary and isopleth images; place the ternary points by hand.")
            return
        try:
            corners = detect_frame(load_rgba(self.cal.image_path))
        except Exception as ex:
            if verbose:
                self.info.setText(f"<span style='color:#a00'>Frame not found: {ex}</span>")
            return
        place_from_frame(self.cal, corners)
        self._building = True
        self.mode.setCurrentIndex(0)
        self._building = False
        self._update_labels()
        self._refresh()
        self.info.setText(self.info.text() + "  Frame detected: the points sit on its corners with the axis bounds "
                                             "as values — check the bounds, and drag any point that is off.")

    # -- form <-> calibration -----------------------------------------------------------------------------
    def _load_into_form(self):
        self._building = True
        c = self.cal
        self.kind.setCurrentIndex([k for k, _ in KINDS].index(c.kind))
        for i, cb in enumerate(self.el):
            if i < len(c.elements) and c.elements[i] in self.elements:
                cb.setCurrentText(c.elements[i])
            elif i < len(self.elements):
                cb.setCurrentIndex(i)
        self.t_unit.setCurrentText(c.t_unit)
        self.x_unit.setCurrentText(c.x_unit)
        for w, val in zip(self.b, c.bounds):
            w.setText(f"{val:g}")
        self.mode.setCurrentIndex(0 if c.mode == "axes" else 1)
        self.persp.setChecked(c.perspective)
        self._building = False
        self._update_labels()

    def _update_labels(self):
        k = self.cal.kind
        tern = k == "ternary"
        self.el[2].setVisible(k != "binary")
        self.el_labels[2].setVisible(k != "binary")
        self.mode.setEnabled(not tern)
        self.t_unit.setEnabled(not tern)
        self.persp.setEnabled(self.cal.mode == "points")
        e = self.cal.elements + ["?", "?", "?"]
        if tern:
            self.el_labels[0].setText("bottom-left")
            self.el_labels[1].setText("bottom-right")
            self.el_labels[2].setText("top")
            self.b_labels[0].setText(f"{e[1].capitalize()} ({self.cal.x_unit})")
            self.b_labels[1].setText(f"{e[2].capitalize()} ({self.cal.x_unit})")
            self.table.setHorizontalHeaderLabels(["Point", "Pixel", f"{e[1].capitalize()}", f"{e[2].capitalize()}"])
        else:
            self.el_labels[0].setText("A")
            self.el_labels[1].setText("x axis")
            self.el_labels[2].setText("fixed")
            self.b_labels[0].setText(f"{e[1].capitalize()} ({self.cal.x_unit})")
            self.b_labels[1].setText(f"T ({self.cal.t_unit})")
            self.table.setHorizontalHeaderLabels(["Point", "Pixel", f"{e[1].capitalize()}", "T"])
        self.add_pt.setEnabled(self.cal.mode == "points")
        self.del_pt.setEnabled(self.cal.mode == "points")

    def _read_form(self):
        c = self.cal
        c.kind = self.kind.currentData()
        n = 3 if c.kind != "binary" else 2
        c.elements = [cb.currentText() for cb in self.el[:n]]
        c.t_unit = self.t_unit.currentText()
        c.x_unit = self.x_unit.currentText()
        vals = [_f(w.text()) for w in self.b]
        if all(v is not None for v in vals):
            c.bounds = vals
        c.perspective = self.persp.isChecked()

    def _on_form(self, *a):
        if self._building:
            return
        self._read_form()
        self._update_labels()
        self._refresh()

    def _on_kind(self, *a):
        if self._building:
            return
        self._read_form()
        if self.cal.kind == "ternary":
            self.cal.mode = "points"
            full = 100.0 if self.cal.x_unit.endswith("%") else 1.0
            self.cal.bounds = [0.0, full, 0.0, full]
        else:
            self.cal.bounds = [0.0, 100.0 if self.cal.x_unit.endswith("%") else 1.0, 0.0, 1000.0]
        self.cal.default_points()
        self._load_into_form()
        self._selected = 0
        self._refresh()

    def _on_mode(self, *a):
        if self._building:
            return
        self._read_form()
        self.cal.mode = self.mode.currentData()
        self.cal.default_points()
        self._selected = 0
        self._update_labels()
        self._refresh()

    def _values_from_bounds(self):
        self._read_form()
        placed = [(p.px, p.py) for p in self.cal.points]
        self.cal.default_points()
        for p, (x, y) in zip(self.cal.points, placed):
            p.px, p.py = x, y
        self._refresh()

    # -- table --------------------------------------------------------------------------------------------
    def _point_name(self, i: int, p: CalPoint) -> str:
        return {"x1": "x1 (x axis)", "x2": "x2 (x axis)", "y1": "T1 (T axis)",
                "y2": "T2 (T axis)"}.get(p.role, f"P{i + 1}")

    def _refresh(self):
        self._building = True
        pts = self.cal.points
        self.table.setRowCount(len(pts))
        for i, p in enumerate(pts):
            items = [QTableWidgetItem(self._point_name(i, p)),
                     QTableWidgetItem("—" if not p.placed else f"{p.px:.1f}, {p.py:.1f}"),
                     QTableWidgetItem("" if p.a is None else f"{p.a:g}"),
                     QTableWidgetItem("" if p.b is None else f"{p.b:g}")]
            for j, it in enumerate(items):
                flags = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
                editable = (j == 2 and p.role in ("", "x1", "x2")) or (j == 3 and p.role in ("", "y1", "y2"))
                if editable:
                    flags |= Qt.ItemFlag.ItemIsEditable
                elif j >= 2:
                    it.setText("—")
                    it.setForeground(QBrush(QColor(160, 160, 160)))
                it.setFlags(flags)
                self.table.setItem(i, j, it)
        if pts:
            self._selected = min(self._selected, len(pts) - 1)
            self.table.selectRow(self._selected)
        self._building = False
        self.view.set_markers([(p.px, p.py, self._point_name(i, p), i == self._selected, i)
                               for i, p in enumerate(pts) if p.placed])
        self._update_overlay()

    def _on_table_edit(self, item: QTableWidgetItem):
        if self._building:
            return
        p = self.cal.points[item.row()]
        v = _f(item.text())
        if item.column() == 2:
            p.a = v
        elif item.column() == 3:
            p.b = v
        self._update_overlay()

    def _on_select(self):
        if self._building:
            return
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self._selected = rows[0].row()
            self.view.set_markers([(p.px, p.py, self._point_name(i, p), i == self._selected, i)
                                   for i, p in enumerate(self.cal.points) if p.placed])

    def _on_click(self, x: float, y: float):
        if not self.cal.points:
            return
        p = self.cal.points[self._selected]
        p.px, p.py = x, y
        unplaced = [i for i, q in enumerate(self.cal.points) if not q.placed]
        nxt = [i for i in unplaced if i > self._selected] or unplaced
        if nxt:
            self._selected = nxt[0]
        self._refresh()

    def _on_drag(self, i: int, x: float, y: float):
        if not (0 <= i < len(self.cal.points)):
            return
        self.cal.points[i].px, self.cal.points[i].py = x, y
        self._selected = i
        self._refresh()

    def _add_point(self):
        self.cal.points.append(CalPoint())
        self._selected = len(self.cal.points) - 1
        self._refresh()

    def _del_point(self):
        if len(self.cal.points) > 0:
            del self.cal.points[self._selected]
            self._selected = max(0, self._selected - 1)
            self._refresh()

    # -- overlay ------------------------------------------------------------------------------------------
    def _update_overlay(self):
        self._read_form()
        try:
            M = self.cal.matrix()
            Minv = np.linalg.inv(M)
        except (ValueError, np.linalg.LinAlgError) as ex:
            self.view.set_overlay([])
            self.info.setText(f"<span style='color:#a00'>{ex}</span>")
            self.bb.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
            return
        lines = []
        b0, b1, c0, c1 = self.cal.bounds
        t = np.linspace(0, 1, 40)
        if self.cal.kind == "ternary":
            s = U.scale(self.cal.x_unit)
            # lines of constant B and constant C, restricted to the bounds and the triangle
            for val in np.unique(np.r_[np.arange(0, 1.0001, 0.1) * s, b0, b1]):
                if not (min(b0, b1) - 1e-9 <= val <= max(b0, b1) + 1e-9):
                    continue
                cs = np.linspace(min(c0, c1), min(max(c0, c1), s - val), 40)
                X, Y = U.tri_to_xy(np.full_like(cs, val) / s, cs / s)
                lines.append(np.stack(apply(Minv, X, Y), 1))
            for val in np.unique(np.r_[np.arange(0, 1.0001, 0.1) * s, c0, c1]):
                if not (min(c0, c1) - 1e-9 <= val <= max(c0, c1) + 1e-9):
                    continue
                bs = np.linspace(min(b0, b1), min(max(b0, b1), s - val), 40)
                X, Y = U.tri_to_xy(bs / s, np.full_like(bs, val) / s)
                lines.append(np.stack(apply(Minv, X, Y), 1))
        else:
            for xv in np.linspace(b0, b1, 6):
                lines.append(np.stack(apply(Minv, np.full_like(t, xv), c0 + (c1 - c0) * t), 1))
            for yv in np.linspace(c0, c1, 6):
                lines.append(np.stack(apply(Minv, b0 + (b1 - b0) * t, np.full_like(t, yv)), 1))
        self.view.set_overlay(lines)
        res = self.cal.residuals()
        msg = "Calibration OK."
        if res and max(res) > 0:
            msg += f" Largest point misfit {max(res):.1f} px."
        self.info.setText(msg)
        self.bb.button(QDialogButtonBox.StandardButton.Ok).setEnabled(True)

    def _accept(self):
        self._read_form()
        try:
            self.cal.matrix()
        except ValueError as ex:
            QMessageBox.warning(self, "Calibration", str(ex))
            return
        if len(set(self.cal.elements)) != len(self.cal.elements):
            QMessageBox.warning(self, "Calibration", "Choose different elements.")
            return
        try:
            self.cal.save_sidecar()
        except OSError:
            pass
        self.accept()
