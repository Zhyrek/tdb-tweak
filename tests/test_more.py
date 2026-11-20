"""Several databases in one engine, G-x curves, step plots, fitting ZPF data, ESPEI extra settings."""
import os

import numpy as np
import pytest

from tdb_tweak.engine import Engine, MapRequest
from tdb_tweak.tdbmodel import TDBDocument

ROOT = os.path.dirname(os.path.dirname(__file__))
MCMC = os.path.join(ROOT, "examples", "Cu-Mg", "Cu-Mg-mcmc.tdb")
GEN = os.path.join(ROOT, "examples", "Cu-Mg", "Cu-Mg-generated.tdb")
REQ = MapRequest("binary", ("CU", "MG"), tmin=500, tmax=1500, tstep=10, xstep=0.01)


def test_engine_keeps_several_databases_compiled():
    a, b = TDBDocument.from_file(MCMC), TDBDocument.from_file(GEN)
    e = Engine()
    assert e.set_database(a.compute_text(), list(a.parameter_values())) is True
    ra = e.run(REQ, a.parameter_values())
    assert e.set_database(b.compute_text(), list(b.parameter_values())) is True
    e.run(REQ, b.parameter_values())
    assert e.set_database(a.compute_text(), list(a.parameter_values())) is False      # back without recompiling
    ra2 = e.run(REQ, a.parameter_values())
    assert [round(ys[0], 3) for _, _, ys in ra.invariants] == [round(ys[0], 3) for _, _, ys in ra2.invariants]


@pytest.fixture(scope="module")
def mcmc():
    d = TDBDocument.from_file(MCMC)
    e = Engine()
    e.set_database(d.compute_text(), list(d.parameter_values()))
    return d, e


def test_gx_curves_tangents(mcmc):
    d, e = mcmc
    g = e.gx_curves(REQ, d.parameter_values(), 900.0)
    pairs = {tuple(sorted(p)) for *_, p in g["tangents"]}
    assert ("FCC_A1", "LAVES_C15") in pairs and ("LAVES_C15", "LIQUID") in pairs
    assert len(g["curves"]["CUMG2"][0]) == 1                  # a line compound is one point


def test_step_plot(mcmc):
    d, e = mcmc
    s = e.step(REQ, d.parameter_values(), {"MG": 0.25}, 600, 1300, n=141)
    liq = s["fractions"]["LIQUID"]
    assert liq[0] == pytest.approx(0, abs=1e-6) and liq[-1] == pytest.approx(1, abs=1e-6)
    total = sum(s["fractions"].values())
    assert np.allclose(total, 1.0, atol=1e-6)


def test_fit_zpf_data_with_artifact_guard():
    from tdb_tweak.artifacts import scan
    from tdb_tweak.optimize import OptSettings, Variable, optimize_invariant
    from tdb_tweak.zpf import load_paths
    ds, _ = load_paths([os.path.join(ROOT, "examples", "Cu-Mg", "zpf")])
    d = TDBDocument.from_file(GEN)
    e = Engine()
    e.set_database(d.compute_text(), list(d.parameter_values()))
    var = [Variable(s, r.basis) for s, r in {x.symbol: x for x in d.rows if x.kind == "VV"}.items()]
    res = optimize_invariant(e, REQ, d.parameter_values(), var, [],
                             OptSettings(fit_zpf=True, zpf_data=ds, max_iter=3, reject_artifacts=True))
    assert res.iterations[0].zpf_rms > 1300 and res.iterations[-1].zpf_rms < 400
    vals = d.parameter_values()
    vals.update(res.iterations[-1].values)
    assert not [f for f in scan(e, ["CU", "MG"], vals) if f.severity == "warning"]


def test_espei_extra_settings():
    from tdb_tweak.espei_polish import parse_extra
    assert parse_extra("backend: cg\ncores: 4  # comment\nprior: {\"name\": \"zero\"}\n") == \
        {"backend": "cg", "cores": 4, "prior": {"name": "zero"}}
