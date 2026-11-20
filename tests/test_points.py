import warnings

import numpy as np
import pytest
from pycalphad import Database
from pycalphad.core.utils import instantiate_models

from tdb_tweak.points import composition_space, constrain

TDB = """ELEMENT VA VACUUM 0 0 0 !
ELEMENT CR BCC_A2 52 0 0 !
ELEMENT FE BCC_A2 55.8 0 0 !
ELEMENT NI FCC_A1 58.7 0 0 !
PHASE LINE % 2 1 2 !
CONSTITUENT LINE :FE:NI,CR: !
PHASE STOI % 2 1 2 !
CONSTITUENT STOI :FE:NI: !
PHASE BCC % 2 1 3 !
CONSTITUENT BCC :CR,FE,NI:VA: !
PARAMETER G(LINE,FE:NI;0) 1 -1000; 10000 N !
PARAMETER G(LINE,FE:CR;0) 1 -1000; 10000 N !
PARAMETER G(STOI,FE:NI;0) 1 -1000; 10000 N !
PARAMETER G(BCC,FE:VA;0) 1 -1000; 10000 N !
"""


@pytest.fixture(scope="module")
def models():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        db = Database.from_string(TDB, fmt="tdb")
        return instantiate_models(db, ["CR", "FE", "NI", "VA"], ["LINE", "STOI", "BCC"])


def test_ranks(models):
    assert composition_space(models["STOI"], ["FE", "NI"])["rank"] == 0
    assert composition_space(models["BCC"], ["FE", "NI"])["rank"] == 2
    line = composition_space(models["LINE"], ["FE", "NI"])
    assert line["rank"] == 1
    assert abs(line["dir"][0]) < 1e-9                       # x(Fe) cannot change ...
    assert line["hi"] - line["lo"] == pytest.approx(2 / 3)    # ... while Ni/Cr exchange over 2/3


def test_constrain_projects_onto_the_line(models):
    line = composition_space(models["LINE"], ["FE", "NI"])
    fe, ni = constrain(line, [0.5, 0.2])
    assert fe == pytest.approx(1 / 3) and ni == pytest.approx(0.2)
    fe, ni = constrain(line, [0.1, 0.95])                     # beyond the end member: clipped to x(Ni) = 2/3
    assert fe == pytest.approx(1 / 3) and ni == pytest.approx(2 / 3)
    assert constrain(composition_space(models["STOI"], ["FE", "NI"]), [0.3, 0.3]) is None
