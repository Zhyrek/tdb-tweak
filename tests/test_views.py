from tdb_tweak.engine import MapRequest
from tdb_tweak.views import ResultCache, SavedView, auto_name, request_key, same_diagram, values_stamp

G = {"kind": "binary", "elements": ["CU", "MG"], "t_unit": "°C", "x_unit": "at.%", "tmin_K": 773.15,
     "tmax_K": 1473.15, "tstep": 10.0, "tsec_K": 1000.0, "xmin": 0.0, "xmax": 40.0, "xstep": 1.0, "ymin": 0.0,
     "ymax": 100.0, "fixed": 10.0, "pdens": 500, "cg": False, "tielines": False}


def test_same_diagram_tolerates_unit_roundoff_but_not_real_changes():
    assert same_diagram(G, {**G, "tmin_K": 773.15 + 1e-9})
    assert not same_diagram(G, {**G, "tmax_K": 1400.0})
    assert not same_diagram(G, {**G, "elements": ["MG", "CU"]})
    assert same_diagram(G, {**G, "tsec_K": 5.0})                 # irrelevant for a binary
    t = {**G, "kind": "ternary", "elements": ["AL", "CR", "NI"]}
    assert not same_diagram(t, {**t, "tsec_K": 1001.0})


def test_auto_names():
    assert auto_name(G) == "Cu-Mg 500–1200 °C, 0–40%"
    assert auto_name({**G, "kind": "ternary", "elements": ["AL", "CR", "NI"], "tsec_K": 1273.15,
                      "xmax": 100.0}) == "Al-Cr-Ni 1000 °C"
    assert auto_name(G, zoomed=True).endswith("(zoom)")


def test_request_key_and_cache():
    a = MapRequest("binary", ("CU", "MG"), tmin=773.15, tmax=1473.15)
    b = MapRequest("binary", ("CU", "MG"), tmin=773.15 + 1e-10, tmax=1473.15)
    assert request_key(a) == request_key(b)
    s1 = values_stamp("text", {"VV0000": 1.0})
    s2 = values_stamp("text", {"VV0000": 2.0})
    c = ResultCache(size=2)
    c.put(request_key(a), s1, "old")
    assert c.get(request_key(b), s1) == "old" and c.get(request_key(a), s2) is None
    c.put(request_key(a), s2, "new")
    assert c.latest_for(request_key(a)) == "new"
    c.put(("other",), s1, "x")
    assert c.get(request_key(a), s1) is None                     # evicted (LRU, size 2)


def test_saved_view_roundtrip():
    v = SavedView("x", dict(G), [[15.0, 40.0], [700.0, 1100.0]])
    assert SavedView.from_dict(v.to_dict()) == v
