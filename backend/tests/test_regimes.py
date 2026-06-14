"""Unit tests for the refactor seams: regime selection and the action mask.

These test the new structure directly (in isolation), where the characterization
tests only exercise it transitively.
"""
from antfarm import AntWorld, SimConfig
from antfarm.env import EAT, N_ACTIONS, NOTHING, REPRODUCE
from antfarm.regimes import (
    HomeostaticRegime,
    LogisticRegime,
    NutrientRegime,
    make_regime,
)


def test_regime_selection():
    assert isinstance(make_regime(SimConfig(device="cpu")), HomeostaticRegime)
    assert isinstance(make_regime(SimConfig(device="cpu", ecosystem=True)), LogisticRegime)
    assert isinstance(
        make_regime(SimConfig(device="cpu", ecosystem=True, food_model="nutrient")),
        NutrientRegime,
    )


def test_env_uses_matching_regime():
    e = AntWorld(SimConfig(device="cpu", ecosystem=True, food_model="nutrient",
                           n_ants=5, max_ants=20, world_size=10))
    assert e.regime.name == "nutrient"


def test_action_mask_shape_and_dtype():
    e = AntWorld(SimConfig(device="cpu", n_ants=6, world_size=10))
    m = e.action_mask()
    assert m.shape == (1, e.n_slots, N_ACTIONS)
    assert m.dtype.is_floating_point is False  # bool


def test_eat_legal_only_on_food_with_room():
    e = AntWorld(SimConfig(device="cpu", n_ants=4, world_size=10))
    e.food.zero_()
    cx, cy = int(e.pos[0, 0, 0]), int(e.pos[0, 0, 1])
    e.food[0, cx, cy] = 5.0
    e.energy[0, 0] = 1.0                      # on food + room -> legal
    assert bool(e.action_mask()[0, 0, EAT])
    e.energy[0, 0] = e.cfg.energy_max         # full -> illegal
    assert not bool(e.action_mask()[0, 0, EAT])
    e.energy[0, 0] = 1.0
    e.food.zero_()                            # no food -> illegal
    assert not bool(e.action_mask()[0, 0, EAT])


def test_reproduce_mask_only_when_ecosystem_and_eligible():
    h = AntWorld(SimConfig(device="cpu", n_ants=5, world_size=10))
    assert not bool(h.action_mask()[..., REPRODUCE].any())   # homeostatic: never

    e = AntWorld(SimConfig(device="cpu", ecosystem=True, n_ants=5, max_ants=20,
                           world_size=10, birth_threshold=0.6))
    e.energy[0, 0] = e.cfg.energy_max         # eligible
    e.energy[0, 1] = 1.0                      # not eligible
    m = e.action_mask()
    assert bool(m[0, 0, REPRODUCE])
    assert not bool(m[0, 1, REPRODUCE])


def test_dead_slots_may_only_do_nothing():
    e = AntWorld(SimConfig(device="cpu", ecosystem=True, n_ants=2, max_ants=6, world_size=10))
    m = e.action_mask()
    assert not bool(e.alive[0, 5])            # slot 5 is an empty pool slot
    assert bool(m[0, 5, NOTHING])
    assert int(m[0, 5].sum()) == 1            # ONLY nothing is legal
