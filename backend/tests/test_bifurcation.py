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
    # points are scaled to [0, 1]
    assert all(0.0 <= v <= 1.0 for _, v in d["points"])
    assert d["r"][0] < d["r"][-1]
