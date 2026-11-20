import os

import numpy as np
import pytest

from tdb_tweak.engine import Engine, MapRequest
from tdb_tweak.optimize import (DEFAULT_FLOORS, InvariantTarget, OptSettings, Variable, limit_for,
                                optimize_invariant)
from tdb_tweak.tdbmodel import TDBDocument

CUMG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "Cu-Mg", "Cu-Mg-mcmc.tdb")
REQ = MapRequest("binary", ("CU", "MG"), tmin=500, tmax=1500, tstep=10, xstep=0.01)
PHASES = ("LAVES_C15", "LIQUID", "FCC_A1")


@pytest.fixture(scope="module")
def setup():
    d = TDBDocument.from_file(CUMG)
    e = Engine()
    e.set_database(d.compute_text(), list(d.parameter_values()))
    seen, variables = set(), []
    for r in d.rows:
        if r.kind == "VV" and r.symbol not in seen:
            seen.add(r.symbol)
            variables.append(Variable(r.symbol, r.basis, r.meaning()))
    return d, e, variables


def _map_invariant(e, values):
    res = e.run(REQ, values)
    return next((xs, ys[0]) for n, xs, ys in res.invariants if sorted(n) == sorted(PHASES))


def test_limit_rule():
    v = Variable("VV0000", "const")
    s = OptSettings()
    assert limit_for(v, 0.001, s) == DEFAULT_FLOORS["const"]          # floor wins for small values
    assert limit_for(v, -2e5, s) == 2e5                                # 100 % of the value wins for large ones
    assert limit_for(Variable("VV0001", "T"), 3.0, s) == 50.0
    assert limit_for(Variable("VV0001", "T", limit_override=2.5), 3.0, s) == 2.5


def test_moves_eutectic_to_target_and_full_map_agrees(setup):
    d, e, variables = setup
    tgt = InvariantTarget(PHASES, 1004.84, T_target=980.0, x_target={"LIQUID": 0.22})
    res = optimize_invariant(e, REQ, d.parameter_values(), variables, tgt, OptSettings(max_iter=10))
    assert res.converged, res.message
    last = res.iterations[-1]
    assert abs(last.T - 980.0) <= 1.0 and abs(last.x["LIQUID"] - 0.22) <= 0.001
    vals = d.parameter_values()
    vals.update(last.values)
    xs, T = _map_invariant(e, vals)                                     # independent check by mapping
    assert T == pytest.approx(last.T, abs=0.05)


def test_per_iteration_limits_are_respected(setup):
    d, e, variables = setup
    s = OptSettings(max_iter=4, rel_limit=0.01, floors={k: v * 1e-3 for k, v in DEFAULT_FLOORS.items()})
    tgt = InvariantTarget(PHASES, 1004.84, T_target=900.0)
    res = optimize_invariant(e, REQ, d.parameter_values(), variables, tgt, s)
    assert not res.converged                                            # far target, tiny steps
    its = res.iterations
    assert len(its) == 5 and its[-1].T < its[0].T                       # moving the right way
    for a, b in zip(its, its[1:]):
        for var in variables:
            assert abs(b.values[var.name] - a.values[var.name]) <= limit_for(var, a.values[var.name], s) * (1 + 1e-9)


def test_congruent_points_detected_and_exact(setup):
    d, e, _ = setup
    res = e.run(REQ, d.parameter_values())
    by = {tuple(sorted(c.phases)): c for c in res.congruent}
    cm, lav = by[("CUMG2", "LIQUID")], by[("LAVES_C15", "LIQUID")]
    assert cm.fixed_x and cm.compound == "CUMG2" and cm.x == pytest.approx(2 / 3)
    assert not lav.fixed_x
    # the maximum of a fine map agrees with the direct solution
    fine = MapRequest("binary", ("CU", "MG"), tmin=lav.T - 20, tmax=lav.T + 5, tstep=0.2,
                      xmin=lav.x - 0.02, xmax=lav.x + 0.02, xstep=0.0005)
    rf = e.run(fine, d.parameter_values())
    Tmax = max(float(np.nanmax(b.y)) for b in rf.boundaries if b.phase in lav.phases)
    assert Tmax == pytest.approx(lav.T, abs=0.05)


def test_congruent_optimisation_solution_phase_and_line_compound(setup):
    d, e, variables = setup
    rows = {r.symbol: r for r in d.rows if r.kind == "VV"}
    lav_vars = [v for v in variables if rows[v.name].phase in ("LAVES_C15", "LIQUID")]
    tgt = InvariantTarget(("LAVES_C15", "LIQUID"), 1073.7, T_target=1100.0, x_target={"LAVES_C15": 0.34},
                          kind="congruent", x_guess=0.331)
    res = optimize_invariant(e, REQ, d.parameter_values(), lav_vars, tgt, OptSettings())
    assert res.converged, res.message
    vals = d.parameter_values()
    vals.update(res.iterations[-1].values)
    lav = next(c for c in e.run(REQ, vals).congruent if "LAVES_C15" in c.phases)
    assert lav.T == pytest.approx(1100.0, abs=1.0) and lav.x == pytest.approx(0.34, abs=0.001)

    cm_vars = [v for v in variables if rows[v.name].phase in ("CUMG2", "LIQUID")]
    tgt = InvariantTarget(("CUMG2", "LIQUID"), 841.1, T_target=860.0, x_target={}, kind="congruent", x_guess=2 / 3)
    res = optimize_invariant(e, REQ, d.parameter_values(), cm_vars, tgt, OptSettings())
    assert res.converged and all(abs(x - 2 / 3) < 1e-9 for it in res.iterations for x in it.x.values())


def _vars(d, variables, phases):
    rows = {r.symbol: r for r in d.rows if r.kind == "VV"}
    return [v for v in variables if rows[v.name].phase in phases]


def test_multiple_targets_hold_others_in_place(setup):
    d, e, variables = setup
    var = _vars(d, variables, ("LAVES_C15", "LIQUID", "CUMG2"))
    base = e.run(REQ, d.parameter_values())
    cm0 = next(c for c in base.congruent if "CUMG2" in c.phases)
    eut0 = next(ys[0] for n, xs, ys in base.invariants if "HCP_A3" in n)
    move = InvariantTarget(("LAVES_C15", "LIQUID"), 1073.7, T_target=1100.0, x_target={"LAVES_C15": 0.34},
                           kind="congruent", x_guess=0.331)
    hold1 = InvariantTarget(("CUMG2", "LIQUID"), cm0.T, T_target=cm0.T, x_target={}, kind="congruent", x_guess=2 / 3)
    eut_names = next(n for n, xs, ys in base.invariants if "HCP_A3" in n)
    eut_x = next(xs for n, xs, ys in base.invariants if "HCP_A3" in n)
    hold2 = InvariantTarget(tuple(eut_names), eut0, T_target=eut0, x_target=dict(zip(eut_names, eut_x)))
    res = optimize_invariant(e, REQ, d.parameter_values(), var, [move, hold1, hold2], OptSettings(max_iter=15))
    assert res.converged, res.message
    assert len(res.iterations[-1].points) == 3
    vals = d.parameter_values()
    vals.update(res.iterations[-1].values)
    after = e.run(REQ, vals)
    lav = next(c for c in after.congruent if "LAVES_C15" in c.phases)
    cm = next(c for c in after.congruent if "CUMG2" in c.phases)
    eut = next(ys[0] for n, xs, ys in after.invariants if "HCP_A3" in n)
    assert lav.T == pytest.approx(1100.0, abs=1.0) and lav.x == pytest.approx(0.34, abs=0.001)
    assert cm.T == pytest.approx(cm0.T, abs=1.0) and eut == pytest.approx(eut0, abs=1.0)


def test_norm_term_shrinks_parameters(setup):
    d, e, variables = setup
    var = _vars(d, variables, ("LAVES_C15", "LIQUID"))
    tgt = InvariantTarget(("LAVES_C15", "LIQUID"), 1073.7, T_target=1100.0, x_target={"LAVES_C15": 0.34},
                          kind="congruent", x_guess=0.331)
    plain = optimize_invariant(e, REQ, d.parameter_values(), var, tgt, OptSettings(max_iter=20))
    normed = optimize_invariant(e, REQ, d.parameter_values(), var, tgt,
                                OptSettings(max_iter=20, norm_weight=0.05, norm_center="zero"))
    assert plain.converged and normed.converged
    assert normed.iterations[-1].norm < 0.6 * plain.iterations[-1].norm
    assert normed.iterations[-1].residual <= 1.0


def test_tie_triangle_corner_moves_others_held():
    alcrni = os.path.join(os.path.dirname(__file__), "alcrni.tdb")
    d = TDBDocument.from_file(alcrni)
    rows = [r for r in d.rows if r.kind == "COEF" and r.phase in ("LIQUID", "B2", "BCC_A2") and len(r.elements) >= 2]
    for r in rows:
        d.live.add(r.rid)
    e = Engine()
    e.set_database(d.compute_text(), list(d.parameter_values()))
    req = MapRequest("ternary", ("AL", "CR", "NI"), temperature=1273, xstep=0.02)
    r0 = e.run(req, d.parameter_values())
    names, xb, xc = next((n, xs, ys) for n, xs, ys in r0.invariants if "LIQUID" in n)
    cur = {n: (b, c) for n, b, c in zip(names, xb, xc)}
    want = {}
    for n, (b, c) in cur.items():
        shift = 0.015 if n == "LIQUID" else 0.0
        want[f"{n}|CR"], want[f"{n}|NI"] = b + shift, c - shift
    tgt = InvariantTarget(tuple(names), 1273, x_target=want, kind="tietriangle", corners_guess=cur)
    res = optimize_invariant(e, req, d.parameter_values(), [Variable(r.symbol, r.basis) for r in rows], tgt,
                             OptSettings(max_iter=10))
    assert res.converged, res.message
    vals = d.parameter_values()
    vals.update(res.iterations[-1].values)
    n2, xb2, xc2 = next((n, xs, ys) for n, xs, ys in e.run(req, vals).invariants if sorted(n) == sorted(names))
    got = {n: (b, c) for n, b, c in zip(n2, xb2, xc2)}
    for n in names:
        assert got[n][0] == pytest.approx(want[f"{n}|CR"], abs=0.0015)
        assert got[n][1] == pytest.approx(want[f"{n}|NI"], abs=0.0015)


def test_invariant_from_compositions_when_the_map_lacks_it(setup):
    """With a T guess far from the mapped node, the start is built from the phase compositions instead."""
    d, e, variables = setup
    base = e.run(REQ, d.parameter_values())
    names, xs, ys = next((n, xs, ys) for n, xs, ys in base.invariants if sorted(n) == sorted(PHASES))
    guess_T = float(ys[0]) + 40.0                       # > 20 K off: the map's node is not used
    tgt = InvariantTarget(tuple(names), guess_T, T_target=980.0, x_target={"LIQUID": 0.22},
                          corners_guess={n: (x,) for n, x in zip(names, xs)})
    res = optimize_invariant(e, REQ, d.parameter_values(), variables, tgt, OptSettings(max_iter=10))
    assert res.iterations[0].T == pytest.approx(float(ys[0]), abs=0.05)   # it found the same invariant
    assert res.converged, res.message
