"""The parallel-worlds sweep returns a well-formed, scaled diagram."""
from antfarm import SimConfig, bifurcation_sweep


def test_sweep_shape_and_ranges():
    base = SimConfig(device="cpu").to_dict()
    # tiny sweep so the test is fast on CPU
    d = bifurcation_sweep(base, r_min=0.3, r_max=3.0, n_r=24,
                          transient=40, sample=30, max_ants=400)
    assert len(d["r"]) == 24
    assert len(d["min"]) == len(d["max"]) == len(d["mean"]) == 24
    assert d["pop_ref"] >= 1.0
    assert all(0.0 <= v <= 1.0 for _, v in d["points"])
    assert d["r"][0] < d["r"][-1]


def test_sweep_with_replicates_and_window():
    # A zoomed r-window with replicates: still n_r distinct r values, but more
    # samples per r (time x replicates).
    base = SimConfig(device="cpu").to_dict()
    d = bifurcation_sweep(base, r_min=0.5, r_max=0.9, n_r=10,
                          transient=30, sample=20, runs_per_r=4, max_ants=400)
    assert d["runs_per_r"] == 4
    assert len(d["r"]) == 10
    assert 0.5 <= d["r"][0] and d["r"][-1] <= 0.9
    # points scaled, and there are clearly more than one per r on average
    assert all(0.0 <= v <= 1.0 for _, v in d["points"])
    assert len(d["points"]) >= 10
