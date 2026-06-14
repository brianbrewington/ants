"""Reward functions for the RL lessons.

The single biggest gap in the 1997 project: DEATH carried no penalty. Ants were
rewarded for eating but never punished for starving, so the learned policy had no
explicit pressure to *survive* — only to nibble when food happened to be under
it. These functions fix that, and set up the team reward of Lesson 3.

All rewards are per-ant tensors, shape [n_worlds, n_slots], aligned with the env.
"""

from __future__ import annotations

import torch

from ..env import AntWorld


def survival_food_reward(env: AntWorld, prev_energy: torch.Tensor,
                         prev_alive: torch.Tensor | None = None, *,
                         food_weight: float = 1.0, step_cost: float = 0.0,
                         death_penalty: float = 10.0) -> torch.Tensor:
    """Reward = energy gained this step (i.e. food eaten net of metabolism),
    minus an optional per-step cost, with a large negative hit for dying.

    Call AFTER env.step(), passing the energy (and optionally the alive mask)
    captured BEFORE the step:

        prev_e = env.energy.clone(); prev_a = env.alive.clone()
        env.step(actions)
        r = survival_food_reward(env, prev_e, prev_a)
    """
    # REVIEW-NOTE 2026-06-14 (deferred): correct dead/empty-slot masking requires
    # the caller to pass prev_alive. This is an RL stub; the training loop in
    # Lesson 1 will always pass it. See docs/review-responses/2026-06-14-gpt-5.5.md.
    reward = food_weight * (env.energy - prev_energy) - step_cost
    if prev_alive is not None:
        died = prev_alive & (~env.alive)
        reward = torch.where(died, torch.full_like(reward, -death_penalty), reward)
        reward = torch.where(prev_alive, reward, torch.zeros_like(reward))  # ignore empty slots
    return reward


def team_reward(env: AntWorld, individual: torch.Tensor) -> torch.Tensor:
    """LESSON 3 STUB — credit assignment for cooperation.

    The altruism question needs reward to accrue to the *group*, not just the
    individual who acted (otherwise self-interest drives out communication, as it
    did in 1997). The plan: share each ant's reward across its local
    neighbourhood (the same colocation neighbourhood the env already computes), or
    decompose a team return per VDN/QMIX. Filled in when we reach Lesson 3.
    """
    raise NotImplementedError("team_reward: implemented in Lesson 3 (CTDE / VDN).")
