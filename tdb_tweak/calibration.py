"""Calibration of a literature phase-diagram image: pixel <-> data coordinates.

Data coordinates are in the image's own units (e.g. wt.% Mg and °C): the image is warped into those axes and
the computed diagram is converted to them, so the two are drawn on the same axes.

Modes
-----
* ``axes`` (binary / isopleth): two points on the composition axis with known compositions and two points on
  the temperature axis with known temperatures.  The points can be anywhere along each axis (not only the
  frame corners), and the axes may be rotated or skewed: each data value is constant along the *other* axis'
  direction.
* ``points``: N >= 3 points with both coordinates known (binary: composition, T; ternary: two compositions).
  3 points give an affine map; with ``perspective`` and N >= 4 a projective map (homography), which also
  corrects a photographed page.

A ternary is drawn in Cartesian triangle coordinates (A bottom-left, B bottom-right, C top), so the
calibration maps pixels to (X_B, X_C) and then to that plane; partial triangles work like whole ones.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

from . import units as U


def _cross(a, b) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def axes_matrix(X1, X2, Y1, Y2, x1: float, x2: float, y1: float, y2: float) -> np.ndarray:
    """3x3 affine matrix pixel -> data from two points on each axis."""
    X1, X2, Y1, Y2 = (np.asarray(p, dtype=float) for p in (X1, X2, Y1, Y2))
    dX, dY = X2 - X1, Y2 - Y1
    den_x = _cross(dX, dY)
    den_y = _cross(dY, dX)
    if abs(den_x) < 1e-9:
        raise ValueError("the composition and temperature axis points are parallel")
    if abs(x2 - x1) == 0 or abs(y2 - y1) == 0:
        raise ValueError("the two points on an axis need different values")
    k = (x2 - x1) / den_x
    m = (y2 - y1) / den_y
    return np.array([[k * dY[1], -k * dY[0], x1 - k * _cross(X1, dY)],
                     [m * dX[1], -m * dX[0], y1 - m * _cross(Y1, dX)],
                     [0.0, 0.0, 1.0]])


def affine_fit(pix: np.ndarray, dat: np.ndarray) -> np.ndarray:
    pix, dat = np.asarray(pix, float), np.asarray(dat, float)
    if len(pix) < 3:
        raise ValueError("an affine calibration needs at least 3 points")
    A = np.hstack([pix, np.ones((len(pix), 1))])
    sol, *_ = np.linalg.lstsq(A, dat, rcond=None)
    if np.linalg.matrix_rank(A) < 3:
        raise ValueError("the calibration points are collinear")
    M = np.eye(3)
    M[:2, :] = sol.T
    return M


def homography_fit(pix: np.ndarray, dat: np.ndarray) -> np.ndarray:
    """Normalised DLT, least squares for N >= 4."""
    pix, dat = np.asarray(pix, float), np.asarray(dat, float)
    if len(pix) < 4:
        raise ValueError("a perspective calibration needs at least 4 points")

    def norm(p):
        c = p.mean(0)
        s = np.sqrt(2) / max(np.mean(np.linalg.norm(p - c, axis=1)), 1e-12)
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])

    Tp, Td = norm(pix), norm(dat)
    ph = (Tp @ np.hstack([pix, np.ones((len(pix), 1))]).T).T
    dh = (Td @ np.hstack([dat, np.ones((len(dat), 1))]).T).T
    rows = []
    for (x, y, _), (u, v, _) in zip(ph, dh):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, s, Vt = np.linalg.svd(np.asarray(rows))
    H = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(Td) @ H @ Tp
    if abs(H[2, 2]) < 1e-15:
        raise ValueError("degenerate calibration points")
    return H / H[2, 2]


def apply(M: np.ndarray, x, y) -> Tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = M[2, 0] * x + M[2, 1] * y + M[2, 2]
    return (M[0, 0] * x + M[0, 1] * y + M[0, 2]) / w, (M[1, 0] * x + M[1, 1] * y + M[1, 2]) / w


@dataclass
class CalPoint:
    px: Optional[float] = None
    py: Optional[float] = None
    a: Optional[float] = None       # binary: composition; ternary: X_B (image units)
    b: Optional[float] = None       # binary: temperature; ternary: X_C (image units)
    role: str = ""                  # axes mode: "x1", "x2", "y1", "y2"

    @property
    def placed(self) -> bool:
        return self.px is not None and self.py is not None


@dataclass
class Calibration:
    image_path: str
    kind: str = "binary"                    # binary / ternary / isopleth
    mode: str = "axes"                      # axes / points
    perspective: bool = False
    t_unit: str = "°C"
    x_unit: str = "at.%"
    elements: List[str] = field(default_factory=list)   # binary/isopleth: [A, B] with x = B; ternary [A, B, C]
    # display bounds in image units: binary (xmin, xmax, tmin, tmax); ternary (b_min, b_max, c_min, c_max)
    bounds: List[float] = field(default_factory=lambda: [0.0, 100.0, 0.0, 1000.0])
    points: List[CalPoint] = field(default_factory=list)
    alpha: float = 0.6
    visible: bool = True

    # -- persistence ---------------------------------------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["image_path"] = self.image_path
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Calibration":
        d = dict(d)
        d["points"] = [CalPoint(**p) for p in d.get("points", [])]
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    @staticmethod
    def sidecar_path(image_path: str) -> str:
        return image_path + ".tdbtweak.json"

    def save_sidecar(self) -> str:
        p = self.sidecar_path(self.image_path)
        d = self.to_dict()
        d["image_path"] = os.path.basename(self.image_path)
        with open(p, "w") as f:
            json.dump(d, f, indent=1)
        return p

    @classmethod
    def load_sidecar(cls, image_path: str) -> Optional["Calibration"]:
        p = cls.sidecar_path(image_path)
        if not os.path.exists(p):
            return None
        with open(p) as f:
            d = json.load(f)
        d["image_path"] = image_path
        return cls.from_dict(d)

    # -- maths -------------------------------------------------------------------------------------------
    def default_points(self):
        if self.mode == "axes":
            xmin, xmax, tmin, tmax = self.bounds
            self.points = [CalPoint(a=xmin, role="x1"), CalPoint(a=xmax, role="x2"),
                           CalPoint(b=tmin, role="y1"), CalPoint(b=tmax, role="y2")]
        elif self.kind == "ternary":
            self.points = [CalPoint(a=0.0, b=0.0), CalPoint(a=100.0 if self.x_unit.endswith("%") else 1.0, b=0.0),
                           CalPoint(a=0.0, b=100.0 if self.x_unit.endswith("%") else 1.0)]
        else:
            xmin, xmax, tmin, tmax = self.bounds
            self.points = [CalPoint(a=xmin, b=tmin), CalPoint(a=xmax, b=tmin), CalPoint(a=xmax, b=tmax),
                           CalPoint(a=xmin, b=tmax)]

    def matrix(self) -> np.ndarray:
        """pixel -> data (image units; ternary: Cartesian triangle plane in image composition units)."""
        pts = [p for p in self.points if p.placed]
        if self.mode == "axes" and self.kind != "ternary":
            r = {p.role: p for p in pts}
            missing = [k for k in ("x1", "x2", "y1", "y2") if k not in r]
            if missing:
                raise ValueError("place the points " + ", ".join(missing))
            return axes_matrix((r["x1"].px, r["x1"].py), (r["x2"].px, r["x2"].py),
                               (r["y1"].px, r["y1"].py), (r["y2"].px, r["y2"].py),
                               r["x1"].a, r["x2"].a, r["y1"].b, r["y2"].b)
        pts = [p for p in pts if p.a is not None and p.b is not None]
        pix = np.array([[p.px, p.py] for p in pts], float)
        if self.kind == "ternary":
            s = U.scale(self.x_unit)
            X, Y = U.tri_to_xy(np.array([p.a for p in pts]) / s, np.array([p.b for p in pts]) / s)
            dat = np.stack([X, Y], 1)
        else:
            dat = np.array([[p.a, p.b] for p in pts], float)
        if self.perspective and len(pts) >= 4:
            return homography_fit(pix, dat)
        return affine_fit(pix, dat)

    def residuals(self) -> List[float]:
        """Distance in pixels between each placed point and where its known value maps back to."""
        try:
            M = self.matrix()
        except ValueError:
            return []
        Minv = np.linalg.inv(M)
        out = []
        for p in self.points:
            if not p.placed:
                continue
            if self.mode == "axes" and self.kind != "ternary":
                out.append(0.0)
                continue
            if p.a is None or p.b is None:
                continue
            if self.kind == "ternary":
                s = U.scale(self.x_unit)
                X, Y = U.tri_to_xy(p.a / s, p.b / s)
            else:
                X, Y = p.a, p.b
            qx, qy = apply(Minv, X, Y)
            out.append(float(np.hypot(qx - p.px, qy - p.py)))
        return out

    def extent(self) -> Tuple[float, float, float, float]:
        """Display extent in data coordinates (x0, x1, y0, y1)."""
        b0, b1, c0, c1 = self.bounds
        if self.kind != "ternary":
            return (min(b0, b1), max(b0, b1), min(c0, c1), max(c0, c1))
        s = U.scale(self.x_unit)
        # bounding box of the region b in [b0, b1], c in [c0, c1] of the triangle
        bs = np.array([b0, b1, b0, b1]) / s
        cs = np.array([c0, c0, c1, c1]) / s
        X, Y = U.tri_to_xy(bs, cs)
        return (float(X.min()), float(X.max()), float(Y.min()), float(Y.max()))

    def rectify(self, rgba: np.ndarray, max_px: int = 1400) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """Warp the image onto the data axes; returns (image for imshow origin='lower', extent)."""
        M = self.matrix()
        Minv = np.linalg.inv(M)
        x0, x1, y0, y1 = self.extent()
        h_img, w_img = rgba.shape[:2]
        # output size: keep about the source resolution across the extent
        cx, cy = apply(M, np.array([0, w_img, 0, w_img]), np.array([0, 0, h_img, h_img]))
        span_x = max(np.ptp(cx), 1e-12)
        span_y = max(np.ptp(cy), 1e-12)
        nx = int(np.clip(w_img * (x1 - x0) / span_x, 50, max_px))
        ny = int(np.clip(h_img * (y1 - y0) / span_y, 50, max_px))
        gx, gy = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
        px, py = apply(Minv, gx, gy)
        out = np.zeros((ny, nx, 4), dtype=np.float32)
        src = rgba.astype(np.float32)
        if src.max() > 1.0:
            src = src / 255.0
        if src.shape[2] == 3:
            src = np.concatenate([src, np.ones(src.shape[:2] + (1,), np.float32)], 2)
        coords = [py.ravel(), px.ravel()]
        for ch in range(4):
            out[..., ch] = ndimage.map_coordinates(src[..., ch], coords, order=1, mode="constant",
                                                   cval=0.0).reshape(ny, nx)
        inside = (px >= 0) & (px <= w_img - 1) & (py >= 0) & (py <= h_img - 1)
        out[..., 3] *= inside
        return out, (x0, x1, y0, y1)


# ---------------------------------------------------------------------------------------------------------
# frame detection (pre-placing the calibration points)
# ---------------------------------------------------------------------------------------------------------
def _dark_mask(rgba: np.ndarray) -> np.ndarray:
    img = rgba.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    gray = img[..., :3].mean(axis=2)
    if img.shape[2] == 4:
        gray = gray * img[..., 3] + (1.0 - img[..., 3])          # transparent = paper
    bg, ink = float(np.median(gray)), float(np.percentile(gray, 0.5))
    return gray < ink + 0.5 * (bg - ink)


def _strongest_lines(xs, ys, angles_deg, horizontal: bool, min_count: float):
    """Best rotation and the positions of long straight lines at it (projection histogram peaks)."""
    best = None
    for a in angles_deg:
        t = np.deg2rad(a)
        # horizontal lines: constant y' = y cos t - x sin t; vertical: constant x' = x cos t + y sin t
        proj = ys * np.cos(t) - xs * np.sin(t) if horizontal else xs * np.cos(t) + ys * np.sin(t)
        lo = np.floor(proj.min())
        h = np.bincount((proj - lo).astype(int))
        score = h.max()
        if best is None or score > best[0]:
            best = (score, a, h, lo)
    _, a, h, lo = best
    peaks = [i for i in range(len(h)) if h[i] >= min_count and h[i] == h[max(0, i - 3):i + 4].max()]
    return a, [p + lo + 0.5 for p in peaks]


def _fit_line(xs, ys, a_deg, pos, horizontal: bool):
    """Least-squares line through the dark pixels within 2 px of a detected line: returns (point, direction)."""
    t = np.deg2rad(a_deg)
    proj = ys * np.cos(t) - xs * np.sin(t) if horizontal else xs * np.cos(t) + ys * np.sin(t)
    sel = np.abs(proj - pos) <= 2.0
    X, Y = xs[sel].astype(float), ys[sel].astype(float)
    if horizontal:
        k, b = np.polyfit(X, Y, 1)
        return np.array([0.0, b]), np.array([1.0, k]) / np.hypot(1.0, k)
    k, b = np.polyfit(Y, X, 1)
    return np.array([b, 0.0]), np.array([k, 1.0]) / np.hypot(1.0, k)


def _intersect(p1, d1, p2, d2):
    A = np.array([d1, -d2]).T
    s = np.linalg.solve(A, p2 - p1)
    return p1 + s[0] * d1


def detect_frame(rgba: np.ndarray, max_deg: float = 6.0) -> dict:
    """Corners of the plot frame (pixels): bottom-left, bottom-right, top-left, top-right.

    Works with a slightly rotated or sheared image: horizontal and vertical frame lines get their own angles."""
    dark = _dark_mask(rgba)
    H, W = dark.shape
    ys, xs = np.nonzero(dark)
    if len(xs) < 50:
        raise ValueError("the image has almost no dark pixels")
    angles = np.arange(-max_deg, max_deg + 1e-9, 0.1)
    ah, hpos = _strongest_lines(xs, ys, angles, True, 0.3 * W)
    av, vpos = _strongest_lines(xs, ys, angles, False, 0.3 * H)
    if not hpos or not vpos:
        raise ValueError("no long straight frame lines found")
    bottom, top = max(hpos), min(hpos)
    left, right = min(vpos), max(vpos)
    lines = {}
    for name, pos, horiz, a in (("bottom", bottom, True, ah), ("top", top, True, ah),
                                ("left", left, False, av), ("right", right, False, av)):
        lines[name] = _fit_line(xs, ys, a, pos, horiz)
    corners = {
        "bottom_left": _intersect(*lines["bottom"], *lines["left"]),
        "bottom_right": _intersect(*lines["bottom"], *lines["right"]),
        "top_left": _intersect(*lines["top"], *lines["left"]),
        "top_right": _intersect(*lines["top"], *lines["right"]),
    }
    bl, br, tl = corners["bottom_left"], corners["bottom_right"], corners["top_left"]
    if np.linalg.norm(br - bl) < 0.2 * W or np.linalg.norm(tl - bl) < 0.2 * H:
        raise ValueError("the detected frame is implausibly small")
    return {k: (float(v[0]), float(v[1])) for k, v in corners.items()}


def place_from_frame(cal: "Calibration", corners: dict) -> None:
    """Axis-points calibration with the frame corners at the axis bounds (binary / isopleth)."""
    xmin, xmax, tmin, tmax = cal.bounds
    bl, br, tl = corners["bottom_left"], corners["bottom_right"], corners["top_left"]
    cal.mode = "axes"
    cal.points = [CalPoint(*bl, a=xmin, role="x1"), CalPoint(*br, a=xmax, role="x2"),
                  CalPoint(*bl, b=tmin, role="y1"), CalPoint(*tl, b=tmax, role="y2")]
