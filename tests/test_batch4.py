"""Thermochemical score, misfit per tie-line, live-drag linear model, auto solver, per-atom ranges,
ternary boundary nodes, ESPEI export."""
import os

import numpy as np
import pytest

from tdb_tweak.engine import Engine, MapRequest, cg_available
from tdb_tweak.optimize import InvariantTarget, OptSettings, Variable, linear_step, optimize_invariant
from tdb_tweak.tdbmodel import TDBDocument

ROOT = os.path.dirname(os.path.dirname(__file__))
CUMG = os.path.join(ROOT, "examples", "Cu-Mg")
MCMC = os.path.join(CUMG, "Cu-Mg-mcmc.tdb")
REQ = MapRequest("binary", ("CU", "MG"), tmin=500, tmax=1500, tstep=10, xstep=0.01)


@pytest.fixture(scope="module")
def mcmc():
    d = TDBDocument.from_file(MCMC)
    e = Engine()
    e.set_database(d.compute_text(), list(d.parameter_values()))
    var = [Variable(s, r.basis) for s, r in {x.symbol: x for x in d.rows if x.kind == "VV"}.items()]
    return d, e, var


def test_thermo_score_matches_espei(mcmc):
    from tdb_tweak.thermo import load_paths, thermo_score
    d, e, _ = mcmc
    ds, msgs = load_paths([os.path.join(CUMG, "thermo"), os.path.join(CUMG, "activity")])
    assert not msgs and all(x.excluded is not None for x in ds)          # tags resolved from run_mcmc.yaml
    s = thermo_score(e, REQ, d.parameter_values(), ds)
    assert s.n == 76 and s.rms == pytest.approx(5.14, abs=0.02)          # ESPEI's own residuals: 76 points, 5.14


def test_zpf_misfit_per_tieline(mcmc):
    from tdb_tweak.zpf import load_paths
    from tdb_tweak.zpfscore import zpf_score
    d, e, _ = mcmc
    ds, _ = load_paths([os.path.join(CUMG, "zpf")])
    v = d.parameter_values()
    v["VV0021"] += 6000
    s = zpf_score(e, REQ, v, ds)
    liq = [val for (p, i), val in s.per_tieline.items() if p.endswith("bagnous1978-liquidus.json")]
    assert len(liq) == 9 and np.median(liq) > 500                       # the misfit shows on the liquidus points


def test_live_drag_linear_model(mcmc):
    d, e, var = mcmc
    tgt = InvariantTarget(("LAVES_C15", "LIQUID", "FCC_A1"), 1004.84, T_target=1004.84,
                          x_target={"LIQUID": 0.2047, "FCC_A1": 0.0604, "LAVES_C15": 0.322})
    lin = optimize_invariant(e, REQ, d.parameter_values(), var, tgt,
                             OptSettings(jacobian_only=True, hold_others=True)).linear
    lin["max_total"] = 1.0
    vals, pred = linear_step(lin, {(0, "T"): 1004.84 - 10})
    assert pred[(0, "T")] == pytest.approx(994.84, abs=0.5)
    v = d.parameter_values()
    v.update(vals)
    T = next(ys[0] for n, xs, ys in e.run(REQ, v).invariants if sorted(n) == sorted(tgt.phases))
    assert T == pytest.approx(pred[(0, "T")], abs=2.0)


@pytest.mark.skipif(not cg_available(), reason="pycalphad-cg not installed")
def test_auto_solver(mcmc):
    from dataclasses import replace
    d, e, _ = mcmc
    r1 = e.run(replace(REQ, solver="auto"), d.parameter_values())
    r2 = e.run(replace(REQ, solver="auto"), d.parameter_values())
    assert "s cg vs" in r1.message and r2.message.startswith("auto:") or "auto:" in r2.message


def test_per_atom_ranges():
    from tdb_tweak.artifacts import parameter_findings
    d = TDBDocument.from_file(MCMC)
    assert d.phase_atoms("LAVES_C15") == 3 and d.phase_atoms("LIQUID") == 1
    assert parameter_findings(d, ["CU", "MG"]) == []
    for sym, flagged in (("VV0021", True), ("VV0010", False)):     # -1e6 J/mol: 1 atom (liquid) vs 3 atoms (Laves)
        d2 = TDBDocument.from_file(MCMC)
        d2.set_value(next(r for r in d2.rows if r.symbol == sym), -1e6)
        assert bool(parameter_findings(d2, ["CU", "MG"])) is flagged
    a = TDBDocument.from_file(os.path.join(ROOT, "tests", "alcrni.tdb"))
    assert parameter_findings(a, a.element_list()) == []         # strong but real Al-Ni terms are not flagged


def test_ternary_boundary_node():
    d = TDBDocument.from_file(os.path.join(ROOT, "tests", "alcrni.tdb"))
    rows = [r for r in d.rows if r.kind == "COEF" and r.phase in ("LIQUID", "B2", "BCC_A2", "L12_FCC", "FCC_A1")
            and len(r.elements) >= 2]
    for r in rows:
        d.live.add(r.rid)
    e = Engine()
    e.set_database(d.compute_text(), list(d.parameter_values()))
    req = MapRequest("ternary", ("AL", "CR", "NI"), temperature=1273, xstep=0.02)
    r0 = e.run(req, d.parameter_values())
    b = next(b for b in r0.boundaries if b.phase == "LIQUID" and b.partner == "B2" and len(b.x) > 6)
    k = len(b.x) // 2
    pa, pb = np.array([b.x[k], b.y[k]]), np.array([b.partner_x[k], b.partner_y[k]])
    dvec = (pb - pa) / np.linalg.norm(pb - pa)
    target = pa - 0.01 * dvec
    tgt = InvariantTarget(("LIQUID", "B2"), 1273, x_target={"LIQUID|across": float(target @ dvec)}, kind="boundary",
                          req=req, point=tuple(target), direction=tuple(dvec),
                          corners_guess={"LIQUID": tuple(pa), "B2": tuple(pb)})
    res = optimize_invariant(e, req, d.parameter_values(), [Variable(r.symbol, r.basis) for r in rows], tgt,
                             OptSettings(max_iter=10))
    assert res.converged, res.message
    v = d.parameter_values()
    v.update(res.iterations[-1].values)
    best = 9.0
    for bb in e.run(req, v).boundaries:
        if bb.phase == "LIQUID" and bb.partner == "B2":
            P = np.c_[bb.x, bb.y]
            for i in range(len(P) - 1):
                sv = P[i + 1] - P[i]
                t = np.clip(np.dot(target - P[i], sv) / max(np.dot(sv, sv), 1e-15), 0, 1)
                best = min(best, float(np.linalg.norm(target - (P[i] + t * sv))))
    assert best < 0.001                                            # the boundary passes through the node


def test_espei_export_and_marks(tmp_path):
    from tdb_tweak.espei_export import add_mark, dataset_dicts, marked_to_dict, write
    from tdb_tweak.zpf import ZPFDataset, load_dataset
    entry = dict(kind="invariant", names=["LAVES_C15", "LIQUID", "FCC_A1"], elements=["CU", "MG"], T=1004.8,
                 want_T=995.0, cur_x={"LAVES_C15": 0.322, "LIQUID": 0.205, "FCC_A1": 0.06},
                 want_x={"LIQUID": 0.21, "FCC_A1": 0.06})
    d = dataset_dicts([entry])[("CU", "MG")]
    ds = load_dataset(write(str(tmp_path / "t.json"), d))
    assert ds.tielines[0].T == 995.0 and {v.phase: v.fractions["MG"] for v in ds.tielines[0].vertices} == \
        {"LAVES_C15": 0.322, "LIQUID": 0.21, "FCC_A1": 0.06}
    m = ZPFDataset("marks", ["CU", "MG"], [], "Marked points", "")
    add_mark(m, 1150.0, "LIQUID", {"MG": 0.1}, "FCC_A1")
    back = load_dataset(write(str(tmp_path / "m.json"), marked_to_dict(m)))
    assert back.tielines[0].vertices[0].fractions == {"MG": 0.1} and not back.tielines[0].vertices[1].known
