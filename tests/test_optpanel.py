"""Offscreen Qt tests: parameter locks, and optimizer phase freezing (which locks/unlocks a phase's parameters)."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from tdb_tweak.optpanel import OptPanel
from tdb_tweak.paramtable import ParamPanel
from tdb_tweak.tdbmodel import TDBDocument

CUMG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "Cu-Mg", "Cu-Mg-mcmc.tdb")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _setup():
    d = TDBDocument.from_file(CUMG)
    seen, rows = set(), []
    for r in d.rows:
        if r.kind == "VV" and r.symbol not in seen:
            seen.add(r.symbol)
            rows.append(r)
    phases_of = {}
    for r in d.rows:
        phases_of.setdefault(r.symbol, set()).add(r.phase)
    return d, rows, phases_of


def test_phase_freeze_locks_and_survives_refill(app):
    d, rows, phases_of = _setup()
    p = OptPanel()
    p.set_variable_rows(rows, d.parameter_values(), phases_of, doc=d)
    assert len(p.chosen()) == 22
    p._toggle_phase("LIQUID")                                   # untick LIQUID: lock all its parameters
    assert all(d.is_locked(r) for r in d.rows_of_phase("LIQUID"))
    chosen = [r for r, _ in p.chosen()]
    assert len(chosen) == 18 and not any(r.phase == "LIQUID" for r in chosen)
    p.set_variable_rows(rows, d.parameter_values(), phases_of, doc=d)          # refill keeps it
    assert p._phase_cbs["LIQUID"].checkState() == Qt.CheckState.Unchecked and len(p.chosen()) == 18
    p._toggle_phase("LIQUID")
    assert len(p.chosen()) == 22 and not d.locked


def test_table_locks_are_shared_with_the_optimizer(app):
    d, rows, phases_of = _setup()
    table = ParamPanel()
    table.set_document(d, ["CU", "MG"])
    opt = OptPanel()
    opt.set_variable_rows(rows, d.parameter_values(), phases_of, doc=d)
    table.locksChanged.connect(opt.refresh_locks)
    liq = [r for r in table.model.rows if r.phase == "LIQUID"]
    table.set_locked(liq[:1], True)                             # one LIQUID parameter locked in the table
    assert opt._phase_cbs["LIQUID"].checkState() == Qt.CheckState.PartiallyChecked
    i = table.model.rows.index(liq[0])
    old = d.value(liq[0])
    assert table.model.setData(table.model.index(i, 2), "123") is False          # locked: no edit
    table.table.setCurrentIndex(table.model.index(i, 2))
    table.nudge(1)
    assert d.value(liq[0]) == old                                                 # ... and no nudge
    opt._toggle_phase("LIQUID")                                 # partly locked -> lock all
    assert all(d.is_locked(r) for r in d.rows_of_phase("LIQUID"))
