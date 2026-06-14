"""The training loop, as its own object.

`Trainer` owns the per-step transition cycle and the reward; the `Learner` owns
how to act and how to update. Keeping them separate is the seam the architecture
review asked for: a learner is no longer "just a policy you swap in" -- it has a
lifecycle a trainer drives. This is deliberately minimal (online, on-policy-ish
tabular Q); off-policy replay buffers and CTDE critics will extend it in later
lessons, but the loop shape stays:

    s1 = encode(env)
    a  = act(env)              # with exploration + legal-action mask
    prev_energy = env.energy   # remember, for the reward
    env.step(a)
    r  = reward(env, prev_energy, env.last_died)
    s2 = encode(env)
    learner.update(s1, a, r, s2, env)

Run it headless across many parallel worlds -- thousands of transitions per step.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from ..env import AntWorld
from .base import Learner
from .rewards import survival_food_reward


class Trainer:
    def __init__(self, env: AntWorld, learner: Learner,
                 reward_fn: Callable = survival_food_reward):
        self.env = env
        self.learner = learner
        self.reward_fn = reward_fn

    def train_step(self) -> torch.Tensor:
        env, learner = self.env, self.learner
        s1 = learner.encode(env)
        actions = learner.act(env)
        prev_energy = env.energy.clone()
        env.step(actions, light=True)              # skip metric syncs; we only learn
        reward = self.reward_fn(env, prev_energy, env.last_died)
        s2 = learner.encode(env)
        learner.update(s1, actions, reward, s2, env)
        return reward

    def train(self, steps: int) -> None:
        for _ in range(steps):
            self.train_step()
