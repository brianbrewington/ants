"""The Learner interface — a policy that learns.

A Learner is callable as `learner(env) -> actions` (the same contract the
hand-written policies use, so it slots into the server/sim loop unchanged) but
adds `encode()` and `update()` hooks that a training loop drives.

Intended per-step training loop (Lesson 1+):

    s1   = learner.encode(env)               # state, shape [n_worlds, n_slots]
    acts = learner.act(env)                  # actions, with exploration
    prev = env.energy.clone()                # for the reward
    info = env.step(acts)
    s2   = learner.encode(env)
    r    = survival_food_reward(env, prev)   # see rewards.py
    learner.update(s1, acts, r, s2, env)     # learn

Keeping the per-ant batch layout [n_worlds, n_slots] all the way through means a
learner trains on every ant in every parallel world at once — the same
vectorization win the env already exploits ("parallel rollouts").
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from ..env import AntWorld


class Learner(ABC):
    name = "learner"

    @abstractmethod
    def act(self, env: AntWorld) -> torch.Tensor:
        """Return one action per ant, shape [n_worlds, n_slots] (long)."""

    @abstractmethod
    def update(self, s1: torch.Tensor, actions: torch.Tensor, reward: torch.Tensor,
               s2: torch.Tensor, env: AntWorld) -> None:
        """Learn from one transition (s1, a, r, s2)."""

    def encode(self, env: AntWorld) -> torch.Tensor:
        """Map the env's observation to this learner's state representation.

        Default raises — concrete learners decide whether they need it (a tabular
        learner does; a pixel/CNN learner might consume the raw observation)."""
        raise NotImplementedError

    def __call__(self, env: AntWorld) -> torch.Tensor:
        return self.act(env)
