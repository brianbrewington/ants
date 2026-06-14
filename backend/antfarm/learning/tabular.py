"""Lesson 1: tabular Q-learning -- a modern, correct, vectorized redo of the
original qupdate.m.

The whole idea in one line: keep a table Q[state, action] = "how good is taking
this action in this state," and nudge each entry toward what experience says it
should be. The "should be" is the Bellman target:

    target = reward + gamma * max_a' Q[next_state, a']
    Q[s, a] += lr * (target - Q[s, a])          # the TD update

i.e. *the value of acting now is the reward you just got plus the discounted value
of the best thing you can do next.* That's Q-learning (Watkins, 1989) -- exactly
what the 1996 code did, but here it runs over every ant in every parallel world at
once, and acts only on legal actions.
"""

from __future__ import annotations

import torch

from ..env import N_ACTIONS, AntWorld
from .base import Learner


class TabularQLearner(Learner):
    name = "tabular_q"

    def __init__(self, env: AntWorld, *, n_energy_bins: int = 10, gamma: float = 0.9,
                 lr: float = 0.2, epsilon: float = 0.2, epsilon_min: float = 0.02,
                 epsilon_decay: float = 0.999):
        self.device = env.device
        self.n_energy_bins = n_energy_bins
        self.gamma = gamma          # how much the future is worth vs right now
        self.lr = lr                # how fast we move Q toward the target
        self.epsilon = epsilon      # exploration rate (decays over time)
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        # State = (energy bin) x (on food?) x (heard food?) -- compact + legible,
        # the same spirit as the original getstate.m.
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
        q = self.Q[state]                                  # [B, S, A]
        mask = env.action_mask()                           # [B, S, A] bool
        neg = torch.finfo(q.dtype).min

        # greedy: best LEGAL action
        greedy = torch.where(mask, q, torch.full_like(q, neg)).argmax(dim=-1)
        # explore: a uniform random LEGAL action (argmax of masked noise; uses the
        # env generator so runs stay reproducible)
        noise = torch.rand(q.shape, device=self.device, generator=env.gen)
        rand_legal = torch.where(mask, noise, torch.full_like(noise, -1.0)).argmax(dim=-1)
        explore = torch.rand(state.shape, device=self.device, generator=env.gen) < self.epsilon
        return torch.where(explore, rand_legal, greedy)

    def update(self, s1: torch.Tensor, actions: torch.Tensor, reward: torch.Tensor,
               s2: torch.Tensor, env: AntWorld) -> None:
        # Only learn from ants that were alive at s1 (alive now, or died this step).
        valid = (env.alive | env.last_died).reshape(-1)
        max_next = self.Q[s2].max(dim=-1).values                 # max_a' Q[s2,a']
        target = reward + self.gamma * max_next                  # Bellman target
        qflat = self.Q.view(-1)
        idx = (s1 * N_ACTIONS + actions).reshape(-1)[valid]      # (state,action) cells
        td = (target.reshape(-1)[valid] - qflat[idx])            # temporal-difference error

        # Many ants share a (state,action) cell; average their TD errors per cell
        # then take one step (a clean vectorized stand-in for the per-ant loop).
        sums = torch.zeros_like(qflat).scatter_add_(0, idx, td)
        counts = torch.zeros_like(qflat).scatter_add_(0, idx, torch.ones_like(td))
        qflat += self.lr * torch.where(counts > 0, sums / counts, torch.zeros_like(sums))

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
