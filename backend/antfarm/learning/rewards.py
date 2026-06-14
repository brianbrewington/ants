"""Reward functions for the RL lessons.

The single biggest gap in the 1996 project: DEATH carried no penalty. Ants were
rewarded for eating but never punished for starving, so the learned policy had no
explicit pressure to *survive* -- only to nibble when food happened to be under
it. `survival_food_reward` fixes that. `team_reward` (Lesson 3) is the stub for
the cooperation/credit-assignment work.

All rewards are per-ant tensors, shape [n_worlds, n_slots], aligned with the env.
"""

from __future__ import annotations

import torch

from ..env import AntWorld


def survival_food_reward(env: AntWorld, prev_energy: torch.Tensor,
                         died: torch.Tensor, *, food_weight: float = 1.0,
                         death_penalty: float = 5.0) -> torch.Tensor:
    """Reward = energy gained this step, with a big negative hit for dying.

    Call AFTER env.step(), passing the energy captured BEFORE it and the env's
    death mask (`env.last_died`):

        prev_e = env.energy.clone()
        env.step(actions)
        r = survival_food_reward(env, prev_e, env.last_died)

    For ants that died, the raw energy delta is meaningless (a respawn refills the
    tank), so we override it with -death_penalty. For everyone else the reward is
    just the net energy change: food eaten minus the step's metabolic cost. Simple,
    but it now contains the one thing 1996 lacked -- a reason not to die.
    """
    reward = food_weight * (env.energy - prev_energy)
    return torch.where(died, torch.full_like(reward, -death_penalty), reward)


def team_reward(env: AntWorld, individual: torch.Tensor) -> torch.Tensor:
    """LESSON 3 STUB -- credit assignment for cooperation (CTDE / VDN). The
    altruism question needs reward to accrue to the *group*, not just the actor."""
    raise NotImplementedError("team_reward: implemented in Lesson 3 (CTDE / VDN).")
