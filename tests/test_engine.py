"""The compile-once path must give what a full reload of the edited file gives."""
import os
import warnings

import numpy as np
import pytest

from tdb_tweak.engine import Engine, MapRequest, cg_available
from tdb_tweak.tdbmodel import TDBDocument

HERE = os.path.dirname(__file__)
CUMG = os.path.join(os.path.dirname(HERE), "examples", "Cu-Mg", "Cu-Mg-generated.tdb")
ALCRNI = os.path.join(HERE, "alcrni.tdb")
REQ = MapRequest("binary", ("CU", "MG"), tmin=500, tmax=1500, tstep=20, xstep=0.02)


def sig(r):
    return (len(r.boundaries), sum(len(b.x) for b in r.boundaries), len(r.invariants),
            round(float(sum(np.nansum(b.x) + np.nansum(b.y) / 1e4 for b in r.boundaries)), 6))


def engine_for(doc):
    e = Engine()
    e.set_database(doc.compute_text(), list(doc.parameter_values()))
    return e


@pytest.mark.parametrize("use_cg", [False, pytest.param(True, marks=pytest.mark.skipif(
    not cg_available(), reason="pycalphad-cg not installed"))])
def test_vv_edit_in_place_matches_reload(use_cg):
    req = MapRequest(**{**REQ.__dict__, "use_cg": use_cg})
    d = TDBDocument.from_file(CUMG)
    e = engine_for(d)
    first = e.run(req, d.parameter_values())
    row = next(r for r in d.rows if r.symbol == "VV0003")
    assert d.set_value(row, d.value(row) + 3000) is False
    changed = e.run(req, d.parameter_values())          # same engine: no recompile
    ref_doc = TDBDocument(d.render())
    ref = engine_for(ref_doc).run(req, ref_doc.parameter_values())
    assert sig(changed) == sig(ref)
    assert sig(changed) != sig(first)


def test_symbolic_coefficient_energies_match_literal():
    """A coefficient turned into a parameter gives the same energies as the number written in the file.

    (Comparing maps is not robust here: pycalphad's mapping of Al-Cr-Ni at 1273 K changes its line count when
    this coefficient moves by 1e-6 J/mol.)"""
    from pycalphad import Database, calculate
    d = TDBDocument.from_file(ALCRNI)
    row = next(r for r in d.rows if r.kind == "COEF" and r.phase == "LIQUID" and "L0(AL,CR,NI)" in r.meaning())
    assert d.set_value(row, 19000.0) is True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sym = Database.from_string(d.compute_text(), fmt="tdb")
        lit = Database.from_string(d.render(), fmt="tdb")
        for ph in ("LIQUID", "B2", "L12_FCC"):
            a = calculate(sym, ["AL", "CR", "NI", "VA"], ph, T=1273, P=101325, N=1, pdens=30,
                          parameters=d.parameter_values(), output="GM").GM.values.ravel()
            b = calculate(lit, ["AL", "CR", "NI", "VA"], ph, T=1273, P=101325, N=1, pdens=30,
                          output="GM").GM.values.ravel()
            assert np.nanmax(np.abs(a - b)) < 1e-6


def test_ternary_section_runs():
    d = TDBDocument.from_file(ALCRNI)
    r = engine_for(d).run(MapRequest("ternary", ("AL", "CR", "NI"), temperature=1273, xstep=0.04),
                          d.parameter_values())
    assert r.complete and len(r.boundaries) > 5 and r.invariants


def test_ternary_section_with_parameters_maps():
    """Regression: TielineStrategy imports point_from_equilibrium itself; it must be patched too."""
    d = TDBDocument.from_file(ALCRNI)
    row = next(r for r in d.rows if r.kind == "COEF" and r.phase == "LIQUID" and "L0(AL,CR,NI)" in r.meaning())
    d.set_value(row, d.value(row) + 1000)          # now a live parameter
    r = engine_for(d).run(MapRequest("ternary", ("AL", "CR", "NI"), temperature=1273, xstep=0.04),
                          d.parameter_values())
    assert r.complete and r.invariants
