"""Invariants for the vectorized world. Runs on CPU so CI (no GPU) matches local.

These pin down the properties that are easy to break and hard to eyeball:
food conservation, the torus, death/respawn semantics, and the cleverest piece
of code in the env -- the vectorized reproduction slot assignment.
"""
import torch

from antfarm import AntWorld, SimConfig, make_policy
from antfarm.env import EAT, REPRODUCE


def _homeo(**kw):
    base = dict(device="cpu", n_worlds=1, n_ants=40, world_size=24)
    base.update(kw)
    return AntWorld(SimConfig(**base))


def _eco(**kw):
    base = dict(device="cpu", ecosystem=True, n_worlds=1, n_ants=20, max_ants=200,
                world_size=24, food_seed=0.05, food_diffusion=0.0)
    base.update(kw)
    return AntWorld(SimConfig(**base))


def test_positions_stay_on_torus():
    for env in (_homeo(), _eco()):
        pol = make_policy("heuristic")
        for _ in range(60):
            env.step(pol(env))
        W = env.cfg.world_size
        assert bool((env.pos >= 0).all()) and bool((env.pos < W).all())


def test_food_nonnegative_and_energy_capped():
    for env in (_homeo(), _eco()):
        pol = make_policy("heuristic")
        for _ in range(80):
            env.step(pol(env))
        assert bool((env.food >= 0).all()), "food went negative"
        cap = env.cfg.energy_max + 1e-3
        assert bool((env.energy[env.alive] <= cap).all()), "energy exceeded capacity"


def test_homeostatic_population_pinned():
    env = _homeo()
    pol = make_policy("random")
    for _ in range(100):
        info = env.step(pol(env))
    assert info["population"] == env.cfg.n_ants
    assert bool(env.alive.all())


def test_eat_does_not_create_energy():
    # With zero metabolism, total energy gained in one EAT step cannot exceed the
    # food that existed -- eating only transfers, never conjures.
    env = _homeo(energy_cost=0.0, n_ants=60, world_size=8)  # crowd them for contention
    # Keep them ALIVE (energy>0) but with room to eat -- energy 0 would trip the
    # homeostatic respawn and confound the measurement.
    env.energy[:] = 5.0
    food_before = float(env.food.sum())
    e_before = float(env.energy.sum())
    env.step(torch.full((1, env.n_slots), EAT))
    gained = float(env.energy.sum()) - e_before
    assert gained >= -1e-5
    assert gained <= food_before + 1e-4, "ants ate more energy than food existed"
    assert bool((env.food >= 0).all())


def test_eco_death_without_respawn():
    # Starve them: no food, high cost -> population must fall (no auto-respawn).
    env = _eco(n_ants=120, max_ants=120, energy_cost=5.0, food_seed=0.0, food_density=0.0)
    env.food.zero_()
    start = int(env.alive.sum())
    for _ in range(10):
        env.step(make_policy("forage")(env))
    assert int(env.alive.sum()) < start


def test_reproduction_slot_assignment_no_collisions():
    # 5 well-fed parents, only 3 free slots -> exactly 3 distinct births (capped).
    # If two parents collided onto one slot we'd see fewer than 3.
    env = _eco(n_ants=5, max_ants=8, energy_cost=0.0, birth_threshold=0.5,
               birth_cost=0.3, food_seed=0.0)
    env.energy[0, :5] = env.cfg.energy_max  # all eligible
    pop_before = int(env.alive.sum())
    free_before = env.n_slots - pop_before
    env.step(torch.full((1, env.n_slots), REPRODUCE))
    born = int(env.alive.sum()) - pop_before
    assert born == min(5, free_before) == 3
    # never exceed the slot cap, ever
    for _ in range(50):
        env.step(torch.full((1, env.n_slots), REPRODUCE))
    assert int(env.alive.sum()) <= env.n_slots


def test_reproduction_parent_pays_cost():
    env = _eco(n_ants=4, max_ants=20, energy_cost=0.0, birth_threshold=0.5, birth_cost=0.4)
    env.energy[0, :4] = env.cfg.energy_max
    total_before = float(env.energy.sum())
    env.step(torch.full((1, env.n_slots), REPRODUCE))
    # energy is conserved across a birth (parent hands `birth_cost` to the child)
    assert abs(float(env.energy.sum()) - total_before) < 1e-3


def test_seed_is_actually_reproducible():
    # The whole point of the RNG fix: same seed + same policy => identical run.
    def run():
        env = AntWorld(SimConfig(device="cpu", ecosystem=True, n_ants=30, max_ants=300,
                                 world_size=20, seed=7, food_seed=0.05))
        pol = make_policy("heuristic")
        for _ in range(40):
            env.step(pol(env))
        return env

    a, b = run(), run()
    assert torch.equal(a.pos, b.pos)
    assert torch.equal(a.energy, b.energy)
    assert torch.equal(a.food, b.food)
    assert torch.equal(a.alive, b.alive)


def test_action_fractions_sum_to_one():
    env = _eco()
    pol = make_policy("heuristic")
    info = None
    for _ in range(20):
        info = env.step(pol(env))
    fr = sum(v for k, v in info.items() if k.startswith("frac_"))
    assert abs(fr - 1.0) < 1e-4
