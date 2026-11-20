"""Boundary nodes, holding other features, the total-change cap, ZPF score, artifact scan, probe, frame detection."""
import os

import numpy as np
import pytest

from tdb_tweak.engine import Engine, MapRequest
from tdb_tweak.optimize import InvariantTarget, OptSettings, Variable, limit_for, optimize_invariant
from tdb_tweak.tdbmodel import TDBDocument

ROOT = os.path.dirname(os.path.dirname(__file__))
MCMC = os.path.join(ROOT, "examples", "Cu-Mg", "Cu-Mg-mcmc.tdb")
GEN = os.path.join(ROOT, "examples", "Cu-Mg", "Cu-Mg-generated.tdb")
REQ = MapRequest("binary", ("CU", "MG"), tmin=500, tmax=1500, tstep=10, xstep=0.01)


@pytest.fixture(scope="module")
def cumg():
    d = TDBDocument.from_file(MCMC)
    e = Engine()
    e.set_database(d.compute_text(), list(d.parameter_values()))
    seen, var = set(), []
    for r in d.rows:
        if r.kind == "VV" and r.symbol not in seen:
            seen.add(r.symbol)
            var.append(Variable(r.symbol, r.basis))
    return d, e, var


def _liquidus(res, T):
    b = next(b for b in res.boundaries if b.phase == "LIQUID" and b.partner == "FCC_A1"
             and np.nanmin(b.y) < T < np.nanmax(b.y))
    o = np.argsort(b.y)
    return float(np.interp(T, b.y[o], b.x[o])), float(np.interp(T, b.y[o], b.partner_x[o]))


def test_boundary_node(cumg):
    d, e, var = cumg
    xl, xf = _liquidus(e.run(REQ, d.parameter_values()), 1200)
    node = InvariantTarget(("LIQUID", "FCC_A1"), 1200, T_target=1200, x_target={"LIQUID": xl + 0.01},
                           kind="boundary", corners_guess={"LIQUID": (xl,), "FCC_A1": (xf,)})
    res = optimize_invariant(e, REQ, d.parameter_values(), var, node, OptSettings())
    assert res.converged, res.message
    vals = d.parameter_values()
    vals.update(res.iterations[-1].values)
    assert _liquidus(e.run(REQ, vals), 1200)[0] == pytest.approx(xl + 0.01, abs=0.0015)


def test_hold_everything_else(cumg):
    d, e, var = cumg
    before = e.run(REQ, d.parameter_values())
    tgt = InvariantTarget(("LAVES_C15", "LIQUID"), 1073.7, T_target=1100, x_target={"LAVES_C15": 0.34},
                          kind="congruent", x_guess=0.331)
    res = optimize_invariant(e, REQ, d.parameter_values(), var, tgt, OptSettings(hold_others=True, max_iter=20))
    assert len(res.holds) >= 4
    vals = d.parameter_values()
    vals.update(res.iterations[-1].values)
    after = e.run(REQ, vals)
    for (n0, _, y0), (n1, _, y1) in zip(sorted(before.invariants, key=lambda t: sorted(t[0])),
                                        sorted(after.invariants, key=lambda t: sorted(t[0]))):
        assert sorted(n0) == sorted(n1) and abs(y0[0] - y1[0]) < 10.0      # held (5 K tolerance, loose check)
    lav = next(c for c in after.congruent if "LAVES_C15" in c.phases)
    assert lav.T == pytest.approx(1100, abs=2.0)


def test_total_change_cap(cumg):
    d, e, var = cumg
    s = OptSettings(max_iter=30, max_total=1.0, rel_limit=0.01, floors={k: v * 1e-3 for k, v in OptSettings().floors.items()})
    tgt = InvariantTarget(("LAVES_C15", "LIQUID", "FCC_A1"), 1004.8, T_target=900.0)
    res = optimize_invariant(e, REQ, d.parameter_values(), var, tgt, s)
    p0, p1 = res.iterations[0].values, res.iterations[-1].values
    for v_ in var:
        assert abs(p1[v_.name] - p0[v_.name]) <= limit_for(v_, p0[v_.name], s) * (1 + 1e-9)


def test_zpf_score_orders_databases_and_reacts():
    from tdb_tweak.zpf import load_paths
    from tdb_tweak.zpfscore import zpf_score
    ds, _ = load_paths([os.path.join(ROOT, "examples", "Cu-Mg", "zpf")])
    scores = {}
    for p in (MCMC, GEN):
        d = TDBDocument.from_file(p)
        e = Engine()
        e.set_database(d.compute_text(), list(d.parameter_values()))
        scores[p] = zpf_score(e, REQ, d.parameter_values(), ds)
        assert scores[p].n == 530
    assert scores[MCMC].rms < scores[GEN].rms               # the MCMC fit agrees better with the data


def test_artifact_scan(cumg):
    from tdb_tweak.artifacts import scan
    d, e, _ = cumg
    assert scan(e, ["CU", "MG"], d.parameter_values()) == []
    bad = d.parameter_values()
    bad["VV0020"] += 40                                       # liquid L0 strongly repulsive at high T
    texts = " ".join(f.text for f in scan(e, ["CU", "MG"], bad))
    assert "miscibility gap" in texts and "LIQUID" in texts


def test_equilibrium_probe(cumg):
    d, e, _ = cumg
    pts = dict((p, (f, x)) for p, f, x in e.point(REQ, d.parameter_values(), 1200, {"MG": 0.08}))
    assert set(pts) == {"LIQUID", "FCC_A1"}
    xl, _ = _liquidus(e.run(REQ, d.parameter_values()), 1200)
    assert pts["LIQUID"][1]["MG"] == pytest.approx(xl, abs=0.002)


def test_frame_detection_on_sheared_frame():
    from tdb_tweak.calibration import detect_frame
    H, W = 500, 700
    img = np.ones((H, W, 4))
    A = np.array([[np.cos(0.03), -np.sin(0.03) + 0.02], [np.sin(0.03), np.cos(0.03)]])     # rotate + shear
    c = np.array([W / 2, H / 2])
    corners = {"bottom_left": (100, 430), "bottom_right": (620, 430), "top_left": (100, 60), "top_right": (620, 60)}
    warped = {k: A @ (np.array(v) - c) + c for k, v in corners.items()}
    for a, b in (("bottom_left", "bottom_right"), ("top_left", "top_right"), ("bottom_left", "top_left"),
                 ("bottom_right", "top_right")):
        for t in np.linspace(0, 1, 3000):
            x, y = warped[a] + t * (warped[b] - warped[a])
            img[int(round(y)) - 1:int(round(y)) + 2, int(round(x)) - 1:int(round(x)) + 2, :3] = 0
    for t in np.linspace(0, 1, 800):                         # a curve inside, which must not be taken for the frame
        img[int(300 - 150 * np.sin(3 * t)), int(150 + 400 * t), :3] = 0
    got = detect_frame(img)
    for k, v in warped.items():
        assert np.hypot(*(np.array(got[k]) - v)) < 1.5


def test_next_version_path(tmp_path):
    from tdb_tweak.app import MainWindow
    p = tmp_path / "cumg.tdb"
    p.write_text("x")
    assert MainWindow.next_version_path(str(p)).endswith("cumg_v2.tdb")
    (tmp_path / "cumg_v2.tdb").write_text("x")
    assert MainWindow.next_version_path(str(p)).endswith("cumg_v3.tdb")
    assert MainWindow.next_version_path(str(tmp_path / "cumg_v2.tdb")).endswith("cumg_v3.tdb")
