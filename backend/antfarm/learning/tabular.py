"""Lesson 1 (SCAFFOLD): tabular Q-learning — a modern, correct redo of the
original qupdate.m.

`encode()` and `act()` are implemented so the scaffold is concrete and runnable;
`update()` is the deliberate TODO we'll fill together in Lesson 1. The point of
shipping it now is to lock the seam: a finished TabularQLearner is just another
policy (callable(env)->actions) and drops into the server unchanged.
"""

from __future__ import annotations

import torch

from ..env import N_ACTIONS, AntWorld
from .base import Learner


class TabularQLearner(Learner):
    name = "tabular_q"

    def __init__(self, env: AntWorld, *, n_energy_bins: int = 10,
                 gamma: float = 0.9, lr: float = 0.1, epsilon: float = 0.1):
        self.device = env.device
        self.n_energy_bins = n_energy_bins
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        # State = (energy bin) x (on food?) x (heard food?)  -> compact + legible,
        # the same spirit as the original getstate.m encoding.
        self.n_states = n_energy_bins * 2 * 2
        self.Q = torch.zeros(self.n_states, N_ACTIONS, device=self.device)

    def encode(self, env: AntWorld) -> torch.Tensor:
        obs = env.observe()
        ebin = (obs["energy"] / env.cfg.energy_max * self.n_energy_bins).long()
        ebin = ebin.clamp(0, self.n_energy_bins - 1)
        onfood = obs["on_food"].long()
        heard = (obs["heard_food"] > 0).long()
        return (ebin * 2 + onfood) * 2 + heard            # [n_worlds, n_slots]

    def act(self, env: AntWorld) -> torch.Tensor:
        state = self.encode(env)
        q = self.Q[state]                                  # [n_worlds, n_slots, N_ACTIONS]
        greedy = q.argmax(dim=-1)
        rand = torch.randint(0, N_ACTIONS, state.shape, device=self.device, generator=env.gen)
        explore = torch.rand(state.shape, device=self.device, generator=env.gen) < self.epsilon
        return torch.where(explore, rand, greedy)

    def update(self, s1: torch.Tensor, actions: torch.Tensor, reward: torch.Tensor,
               s2: torch.Tensor, env: AntWorld) -> None:
        # TODO(Lesson 1): the Q-learning Bellman update, vectorized over all ants:
        #
        #   target = reward + gamma * max_a' Q[s2, a']
        #   Q[s1, a] += lr * (target - Q[s1, a])
        #
        # Use index_put_/scatter to apply it across the whole [n_worlds, n_slots]
        # batch at once. Decay epsilon over time. Mind dead slots (mask them out).
        raise NotImplementedError("TabularQLearner.update — implemented in Lesson 1.")
