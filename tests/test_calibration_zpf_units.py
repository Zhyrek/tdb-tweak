import os

import numpy as np
import pytest

from tdb_tweak import units as U
from tdb_tweak.calibration import CalPoint, Calibration, affine_fit, apply, axes_matrix, homography_fit
from tdb_tweak.zpf import load_paths, matching, temperatures

ROOT = os.path.dirname(os.path.dirname(__file__))


def _true_affine():
    # data -> pixel: rotated 3 degrees, sheared, y axis pointing down
    th = np.deg2rad(3.0)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    S = np.array([[1, 0.05], [0, 1]])
    K = np.diag([8.0, -0.4])  # 8 px per wt.%, -0.4 px per degree
    A = R @ S @ K
    t = np.array([120.0, 700.0])
    return lambda x, y: A @ np.array([x, y]) + t


def test_axes_mode_with_interior_ticks_and_skew():
    f = _true_affine()
    X1, X2 = f(10, 300), f(50, 300)          # on the x axis (T = 300 is the axis line)
    Y1, Y2 = f(0, 400), f(0, 1000)           # on the T axis (x = 0)
    M = axes_matrix(X1, X2, Y1, Y2, 10, 50, 400, 1000)
    for x, y in [(0, 300), (33.3, 777.0), (60, 1100)]:
        px = f(x, y)
        gx, gy = apply(M, px[0], px[1])
        assert gx == pytest.approx(x, abs=1e-9) and gy == pytest.approx(y, abs=1e-9)


def test_homography_and_affine_fits():
    rng = np.random.default_rng(0)
    H = np.array([[0.9, 0.1, 5.0], [-0.05, 1.1, 2.0], [1e-4, -2e-4, 1.0]])
    dat = rng.uniform(0, 100, size=(8, 2))
    hom = np.hstack([dat, np.ones((8, 1))]) @ np.linalg.inv(H).T
    pix = hom[:, :2] / hom[:, 2:]
    M = homography_fit(pix, dat)
    gx, gy = apply(M, pix[:, 0], pix[:, 1])
    assert np.allclose(np.c_[gx, gy], dat, atol=1e-6)
    A = affine_fit(pix[:3], dat[:3])
    gx, gy = apply(A, pix[:3, 0], pix[:3, 1])
    assert np.allclose(np.c_[gx, gy], dat[:3], atol=1e-6)


def test_ternary_partial_triangle_calibration_and_rectify():
    # image of the Al-rich corner: pixel = 500 * cartesian, y down
    def to_px(b, c):
        x, y = U.tri_to_xy(b / 100, c / 100)
        return 50 + 500 * x, 450 - 500 * y
    cal = Calibration("x.png", kind="ternary", mode="points", x_unit="at.%", elements=["AL", "CR", "NI"],
                      bounds=[0, 40, 0, 40])
    cal.points = [CalPoint(*to_px(b, c), a=b, b=c) for b, c in [(10, 0), (40, 0), (10, 30)]]
    M = cal.matrix()
    gx, gy = apply(M, *to_px(20, 10))
    assert np.allclose([gx, gy], U.tri_to_xy(0.2, 0.1), atol=1e-9)
    img = np.ones((500, 600, 4), np.uint8) * 255
    arr, ext = cal.rectify(img)
    assert arr.shape[2] == 4 and ext[0] == pytest.approx(0) and ext[1] == pytest.approx(0.6)
    assert max(cal.residuals()) < 1e-6


def test_calibration_sidecar_roundtrip(tmp_path):
    p = tmp_path / "fig.png"
    p.write_bytes(b"")
    cal = Calibration(str(p), kind="binary", t_unit="°C", x_unit="wt.%", elements=["CU", "MG"],
                      bounds=[0, 60, 300, 1100])
    cal.default_points()
    cal.points[0].px, cal.points[0].py = 1.0, 2.0
    cal.save_sidecar()
    back = Calibration.load_sidecar(str(p))
    assert back.points[0].px == 1.0 and back.bounds == [0, 60, 300, 1100] and back.image_path == str(p)


def test_units_roundtrip():
    m = {"CU": 63.546, "MG": 24.305, "NI": 58.69}
    x = np.array([0.0, 0.2, 0.7, 1.0])
    w = U.binary_x_to_display(x, "CU", "MG", m, "wt.%")
    assert np.allclose(U.binary_x_from_display(w, "CU", "MG", m, "wt.%"), x)
    assert U.t_from_K(273.15, "°C") == pytest.approx(0)
    assert U.t_to_K(212, "°F") == pytest.approx(373.15)
    fr = U.x_from_display({"MG": 10.0, "NI": 30.0}, ["CU", "MG", "NI"], m, "wt.%")
    back = U.x_to_display(fr, ["CU", "MG", "NI"], m, "wt.%", ["MG", "NI"])
    assert back["MG"] == pytest.approx(10.0) and back["NI"] == pytest.approx(30.0)


def test_zpf_loading():
    ds, msgs = load_paths([os.path.join(ROOT, "examples", "Cu-Mg", "zpf")])
    assert len(ds) == 10 and not msgs
    assert len(matching(ds, ["CU", "MG"])) == 10 and not matching(ds, ["CU", "NI"])
    temps = temperatures(ds, ["CU", "MG"])
    assert temps and all(n > 0 for _, n in temps)
    unknown = [v for d in ds for tl in d.tielines for v in tl.vertices if not v.known]
    assert unknown, "datasets with null compositions are kept (phase present, composition unknown)"
