"""Offscreen Qt test of the optimizer panel's phase freezing."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

from tdb_tweak.optpanel import OptPanel
from tdb_tweak.tdbmodel import TDBDocument

CUMG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "Cu-Mg", "Cu-Mg-mcmc.tdb")


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_freezing_a_phase_removes_its_parameters_and_survives_refill(app):
    d = TDBDocument.from_file(CUMG)
    seen, rows = set(), []
    for r in d.rows:
        if r.kind == "VV" and r.symbol not in seen:
            seen.add(r.symbol)
            rows.append(r)
    phases_of = {}
    for r in d.rows:
        phases_of.setdefault(r.symbol, set()).add(r.phase)
    p = OptPanel()
    p.set_variable_rows(rows, d.parameter_values(), phases_of)
    assert len(p.chosen()) == 22
    p._phase_cbs["LIQUID"].setChecked(False)
    chosen = [r for r, _ in p.chosen()]
    assert len(chosen) == 18 and not any(r.phase == "LIQUID" for r in chosen)
    p.set_variable_rows(rows, d.parameter_values(), phases_of)          # refill keeps the freeze
    assert not p._phase_cbs["LIQUID"].isChecked() and len(p.chosen()) == 18
    p._phase_cbs["LIQUID"].setChecked(True)
    assert len(p.chosen()) == 22
