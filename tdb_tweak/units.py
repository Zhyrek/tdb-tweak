"""Temperature and composition units of diagram axes.

Internally everything is kelvin and mole fractions.  A *display* uses a temperature unit (K, °C, °F) and a
composition unit: mole fraction, at.%, mass fraction or wt.%.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

T_UNITS = ["K", "°C", "°F"]
X_UNITS = ["at.%", "mole fraction", "wt.%", "mass fraction"]


def t_from_K(T, unit: str):
    T = np.asarray(T, dtype=float)
    if unit == "K":
        return T
    if unit == "°C":
        return T - 273.15
    if unit == "°F":
        return (T - 273.15) * 9.0 / 5.0 + 32.0
    raise ValueError(unit)


def t_to_K(T, unit: str):
    T = np.asarray(T, dtype=float)
    if unit == "K":
        return T
    if unit == "°C":
        return T + 273.15
    if unit == "°F":
        return (T - 32.0) * 5.0 / 9.0 + 273.15
    raise ValueError(unit)


def is_mass(unit: str) -> bool:
    return unit in ("wt.%", "mass fraction")


def scale(unit: str) -> float:
    return 100.0 if unit.endswith("%") else 1.0


def x_to_display(fracs: Dict[str, np.ndarray], elements: Sequence[str], masses: Dict[str, float], unit: str,
                 which: Sequence[str]) -> Dict[str, np.ndarray]:
    """Mole fractions of ``elements`` (all of them, they sum to 1) -> composition of ``which`` in ``unit``."""
    if is_mass(unit):
        w = {e: np.asarray(fracs[e], dtype=float) * masses[e] for e in elements}
        tot = sum(w.values())
        out = {e: w[e] / tot for e in which}
    else:
        out = {e: np.asarray(fracs[e], dtype=float) for e in which}
    s = scale(unit)
    return {e: out[e] * s for e in which}


def x_from_display(vals: Dict[str, np.ndarray], elements: Sequence[str], masses: Dict[str, float], unit: str
                   ) -> Dict[str, np.ndarray]:
    """Composition of all but the first element (``vals``, in ``unit``) -> mole fractions of every element."""
    s = scale(unit)
    rest = [e for e in elements if e in vals]
    first = [e for e in elements if e not in vals]
    if len(first) != 1:
        raise ValueError("give every element but one")
    f = {e: np.asarray(vals[e], dtype=float) / s for e in rest}
    f[first[0]] = 1.0 - sum(f.values())
    if is_mass(unit):
        n = {e: f[e] / masses[e] for e in elements}
        tot = sum(n.values())
        return {e: n[e] / tot for e in elements}
    return {e: f[e] for e in elements}


def binary_x_to_display(xB, A: str, B: str, masses, unit: str):
    xB = np.asarray(xB, dtype=float)
    return x_to_display({A: 1 - xB, B: xB}, [A, B], masses, unit, [B])[B]


def binary_x_from_display(val, A: str, B: str, masses, unit: str):
    return x_from_display({B: val}, [A, B], masses, unit)[B]


# ternary Gibbs triangle: A bottom-left (0,0), B bottom-right (1,0), C top (0.5, sqrt(3)/2)
SQ3 = np.sqrt(3.0) / 2.0


def tri_to_xy(b, c):
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    return b + 0.5 * c, SQ3 * c


def xy_to_tri(x, y):
    c = np.asarray(y, dtype=float) / SQ3
    return np.asarray(x, dtype=float) - 0.5 * c, c
