"""SimConfig validation: clamp every client-supplied value to a safe range."""
from antfarm import SimConfig


def test_structural_sizes_clamped():
    c = SimConfig.from_dict(dict(world_size=-5, n_ants=0, n_worlds=0, max_ants=3, device="cpu"))
    assert c.world_size >= 2
    assert c.n_ants >= 1
    assert c.n_worlds >= 1
    # max_ants can never be below the live population
    assert c.max_ants >= c.n_ants


def test_huge_sizes_capped():
    c = SimConfig.from_dict(dict(world_size=10**9, n_ants=10**9, n_worlds=10**9, device="cpu"))
    assert c.world_size <= 1024
    assert c.n_ants <= 200_000
    assert c.n_worlds <= 4096


def test_rate_ranges():
    c = SimConfig.from_dict(dict(food_diffusion=99, food_density=5, birth_threshold=9,
                                 food_growth_rate=99, energy_cost=-1, device="cpu"))
    assert 0.0 <= c.food_diffusion <= 0.25      # PDE stability bound
    assert 0.0 <= c.food_density <= 1.0
    assert 0.0 <= c.birth_threshold <= 1.0
    assert 0.0 <= c.food_growth_rate <= 4.0
    assert c.energy_cost >= 0.0


def test_from_dict_ignores_unknown_keys():
    c = SimConfig.from_dict(dict(world_size=32, bogus_key=123, device="cpu"))
    assert c.world_size == 32
    assert not hasattr(c, "bogus_key")


def test_with_updates_revalidates():
    c = SimConfig(device="cpu").with_updates(food_diffusion=10.0)
    assert c.food_diffusion <= 0.25
