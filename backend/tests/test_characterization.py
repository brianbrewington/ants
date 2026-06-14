"""Characterization (golden) tests — the SAFETY NET for the world-dynamics refactor.

These pin the *observable behavior* of each world model through the PUBLIC
interface only (SimConfig -> AntWorld.step -> info / snapshot), so they survive
internal reorganization (e.g. moving food dynamics into strategy objects). If a
refactor is behavior-preserving these stay green; if it isn't, they fail loudly.

Goldens were captured on CPU with seed=0 and the heuristic policy at commit
8dab90a (pre-refactor). Population is matched exactly; float aggregates within a
small relative tolerance to allow benign op-reordering but catch real drift.
"""
import pytest

from antfarm import AntWorld, SimConfig, make_policy

# mode -> (population, total_food, total_nutrient, mean_energy, pos_sum)
GOLDEN = {
    "homeostatic": (60, 865.31, 0.0, 12.7842, 1328.2),
    "logistic": (400, 10581.56, 0.0, 19.5772, 9482.16),
    "nutrient": (202, 1995.44, 5680.15, 7.8634, 4604.45),
}

CONFIGS = {
    "homeostatic": dict(n_ants=60, world_size=24),
    "logistic": dict(ecosystem=True, food_model="logistic", n_ants=60, max_ants=400,
                     world_size=24, food_seed=0.05, food_diffusion=0.0),
    "nutrient": dict(ecosystem=True, food_model="nutrient", n_ants=60, max_ants=400,
                     world_size=24, nutrient_init=14, germination=0.05, food_density=0.08),
}


def _signature(mode):
    cfg = SimConfig(device="cpu", seed=0, n_worlds=1, **CONFIGS[mode])
    env = AntWorld(cfg)
    pol = make_policy("heuristic")
    info = None
    for _ in range(200):
        info = env.step(pol(env))
    snap = env.snapshot()
    pos_sum = round(sum(x + y for x, y in snap["ants"]["pos"]), 2)
    return (info["population"], round(info["total_food"], 2),
            round(info["total_nutrient"], 2), round(info["mean_energy"], 4), pos_sum)


@pytest.mark.parametrize("mode", list(GOLDEN))
def test_behavior_matches_golden(mode):
    pop, food, nutrient, energy, pos = _signature(mode)
    gpop, gfood, gnut, gen, gpos = GOLDEN[mode]
    assert pop == gpop, f"{mode}: population {pop} != golden {gpop}"

    def close(a, b, rel=2e-3, absol=1e-6):
        return abs(a - b) <= max(absol, rel * abs(b))

    assert close(food, gfood), f"{mode}: total_food {food} != {gfood}"
    assert close(nutrient, gnut), f"{mode}: total_nutrient {nutrient} != {gnut}"
    assert close(energy, gen), f"{mode}: mean_energy {energy} != {gen}"
    assert close(pos, gpos), f"{mode}: pos_sum {pos} != {gpos}"
