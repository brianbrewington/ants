"""Closed nutrient loop (Lesson 0.6) invariants.

The whole point of this model: food grows from a conserved nutrient pool, so
(a) total mass is conserved in a closed system, (b) a zeroed food cell is no
longer absorbing (it germinates again from the nutrient), and (c) high growth
rate no longer drives an absorbing collapse the way the logistic model did.
"""

from antfarm import AntWorld, SimConfig, make_policy


def _nutrient_env(**kw):
    base = dict(device="cpu", ecosystem=True, food_model="nutrient", n_worlds=1,
                n_ants=40, max_ants=400, world_size=24, nutrient_init=14.0,
                germination=0.05, half_sat=3.0, nutrient_inflow=0.0,
                food_diffusion=0.0, energy_cost=0.4, bite_size=2.0,
                food_growth_rate=1.5, birth_threshold=0.6, birth_cost=0.5)
    base.update(kw)
    return AntWorld(SimConfig(**base))


def _total_mass(env) -> float:
    # dead slots have energy 0, so summing all energy == summing live energy
    return float(env.nutrient.sum() + env.food.sum() + env.energy.sum())


def test_closed_system_conserves_mass():
    env = _nutrient_env()
    pol = make_policy("heuristic")
    m0 = _total_mass(env)
    for _ in range(300):
        env.step(pol(env))
    m1 = _total_mass(env)
    assert abs(m1 - m0) / m0 < 1e-3, f"mass drifted: {m0:.3f} -> {m1:.3f}"


def test_inflow_increases_mass():
    # Opening the system (sunlight) should grow the total mass over time.
    env = _nutrient_env(nutrient_inflow=0.5)
    pol = make_policy("heuristic")
    m0 = _total_mass(env)
    for _ in range(100):
        env.step(pol(env))
    assert _total_mass(env) > m0


def test_zero_food_is_not_absorbing():
    env = _nutrient_env()
    pol = make_policy("heuristic")
    for _ in range(50):
        env.step(pol(env))
    env.food.zero_()                       # wipe ALL food
    assert float(env.food.sum()) == 0.0
    for _ in range(60):
        env.step(pol(env))
    assert float(env.food.sum()) > 0.0, "food failed to germinate back from nutrient"


def test_high_r_food_does_not_get_absorbed():
    # The logistic model collapses food to an absorbing zero at high r; the
    # nutrient model must keep food alive (germination from the nutrient pool).
    env = _nutrient_env(food_growth_rate=3.8, n_ants=120, max_ants=2000)
    pol = make_policy("forage")
    for _ in range(500):
        env.step(pol(env))
    assert float(env.food.sum()) > 0.0
    assert float(env.nutrient.sum()) > 0.0
