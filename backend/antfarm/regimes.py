"""
World regimes — the three resource/population models, as strategy objects.

`AntWorld.step()` used to branch on `cfg.ecosystem` / `cfg.food_model` through the
metabolism, births/deaths, and food-growth sections. That entangling is what bred
subtle interaction bugs, so the regime-specific *sequence* now lives here while
the env keeps the shared *mechanics* (eat/move/comm/reproduce + the food-growth
helpers). `step()` is pure orchestration:

    eat / move / communicate          (common, in env)
    regime.metabolize(env, actions)
    n_born, n_dead = regime.population_step(env, actions, light)

Each regime is a thin orchestrator that calls env helpers in a fixed order. The
order of operations (and of RNG draws) is preserved exactly from the pre-refactor
code, so behaviour is identical — the characterization tests are the guard.

Three regimes:
  * Homeostatic — Lesson 0: food refills to a target, dead ants respawn (pinned N).
  * Logistic    — Lesson 0.5: logistic food, free population.
  * Nutrient    — Lesson 0.6: food grows from a conserved nutrient pool; metabolism
                  and death return mass to the soil.
"""

from __future__ import annotations

import torch


class Regime:
    name = "base"

    def reset_resources(self, env) -> None:
        """Initialize env.food / env.nutrient (allocation already done by reset())."""
        raise NotImplementedError

    def metabolize(self, env, actions) -> None:
        """Apply per-step energy cost (and, in the nutrient loop, return it to N)."""
        raise NotImplementedError

    def population_step(self, env, actions, light: bool) -> tuple[int, int]:
        """Reproduce / die / respawn and grow the resource. Returns (n_born, n_dead)."""
        raise NotImplementedError


class HomeostaticRegime(Regime):
    name = "homeostatic"

    def reset_resources(self, env) -> None:
        env._spawn_food(env._target_food_cells(), only_if_under_target=False)

    def metabolize(self, env, actions) -> None:
        env.energy = torch.where(env.alive, env.energy - env.cfg.energy_cost, env.energy)

    def population_step(self, env, actions, light: bool) -> tuple[int, int]:
        dead = env.energy <= 0
        env.last_died = dead                  # death signal for the RL reward
        n_dead = int(dead.sum().item())
        if n_dead:
            env._respawn(dead)
        env._spawn_food(max(1, env._target_food_cells() // 8))
        return 0, n_dead


class LogisticRegime(Regime):
    name = "logistic"

    def reset_resources(self, env) -> None:
        cfg, g = env.cfg, env.gen
        B, W, dev = cfg.n_worlds, cfg.world_size, env.device
        seed = (torch.rand(B, W, W, device=dev, generator=g) < cfg.food_density).float()
        env.food = seed * (0.5 + 0.5 * torch.rand(B, W, W, device=dev, generator=g)) * cfg.max_food_size

    def metabolize(self, env, actions) -> None:
        env.energy = torch.where(env.alive, env.energy - env.cfg.energy_cost, env.energy)

    def population_step(self, env, actions, light: bool) -> tuple[int, int]:
        n_born = env._reproduce(actions, light=light)
        dead = env.alive & (env.energy <= 1e-6)
        env.last_died = dead                  # death signal for the RL reward
        env.alive[dead] = False
        env.energy[dead] = 0.0
        env._grow_food()
        n_dead = 0 if light else int(dead.sum().item())
        return n_born, n_dead


class NutrientRegime(Regime):
    name = "nutrient"

    def reset_resources(self, env) -> None:
        cfg, g = env.cfg, env.gen
        B, W, dev = cfg.n_worlds, cfg.world_size, env.device
        # Most mass starts in the nutrient pool; seed some initial food from it
        # (mass-conserving) so the starting population doesn't starve before
        # germination ramps.
        env.nutrient = torch.full((B, W, W), cfg.nutrient_init, device=dev, dtype=torch.float32)
        seed = (torch.rand(B, W, W, device=dev, generator=g) < cfg.food_density).float()
        sprout = seed * 0.5 * env.nutrient
        env.food = sprout
        env.nutrient = env.nutrient - sprout

    def metabolize(self, env, actions) -> None:
        cfg = env.cfg
        avail = env.energy.clamp(min=0.0)
        burn = torch.where(env.alive,
                           torch.minimum(torch.full_like(avail, cfg.energy_cost), avail),
                           torch.zeros_like(avail))
        env._deposit_nutrient(burn)
        env.energy = env.energy - burn

    def population_step(self, env, actions, light: bool) -> tuple[int, int]:
        n_born = env._reproduce(actions, light=light)
        dead = env.alive & (env.energy <= 1e-6)
        # any body mass left at death returns to the soil (carcass)
        env._deposit_nutrient(torch.where(dead, env.energy.clamp(min=0.0),
                                          torch.zeros_like(env.energy)))
        env.alive[dead] = False
        env.energy[dead] = 0.0
        env._grow_food_nutrient()
        n_dead = 0 if light else int(dead.sum().item())
        return n_born, n_dead


def make_regime(cfg) -> Regime:
    if not cfg.ecosystem:
        return HomeostaticRegime()
    if cfg.food_model == "nutrient":
        return NutrientRegime()
    return LogisticRegime()
