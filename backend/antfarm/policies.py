"""
"Brains" for the ants.

A policy maps the current observation to one action per ant per world. The
contract is intentionally tiny:

    actions = policy(env)          # -> LongTensor [n_worlds, n_ants]

so that in later lessons we can drop in a *learned* policy (a neural net, a
Q-table) behind the exact same interface. For Lesson 0 we ship two hand-written
policies so there's something to watch:

  * RandomPolicy    -- pure noise. Ants rarely eat on purpose, so they starve
                       and respawn constantly. This is the motivation for RL:
                       "we need a brain."
  * HeuristicPolicy -- a sensible rule-of-thumb: eat when on food, sometimes
                       announce food, sometimes go listen. Lively and shows off
                       the communication mechanics. It is NOT learned -- it's the
                       baseline a learned policy will have to beat.
"""

from __future__ import annotations

import torch

from .env import AntWorld, EAT, BROADCAST, NOTHING, TELEPORT, LISTEN, RANDMOVE, N_ACTIONS


class RandomPolicy:
    name = "random"

    def __call__(self, env: AntWorld) -> torch.Tensor:
        B, N = env.cfg.n_worlds, env.cfg.n_ants
        return torch.randint(0, N_ACTIONS, (B, N), device=env.device)


class HeuristicPolicy:
    """Rule-of-thumb baseline.

    Priorities, highest first:
      1. If standing on food and not full -> EAT.
      2. Else, small chance to BROADCAST (more likely if we just ate / are on
         food) to advertise the spot.
      3. Else, small chance to LISTEN for someone else's food, or TELEPORT
         toward a destination we previously heard about.
      4. Otherwise wander (RANDMOVE).
    All decisions are vectorized -- no Python loop over ants.
    """

    name = "heuristic"

    def __init__(self, p_broadcast: float = 0.15, p_listen: float = 0.25,
                 p_teleport: float = 0.5):
        self.p_broadcast = p_broadcast
        self.p_listen = p_listen
        self.p_teleport = p_teleport

    def __call__(self, env: AntWorld) -> torch.Tensor:
        obs = env.observe()
        B, N = env.cfg.n_worlds, env.cfg.n_ants
        dev = env.device
        on_food = obs["on_food"].bool()
        heard_food = obs["heard_food"].bool()

        # Start everyone wandering.
        act = torch.full((B, N), RANDMOVE, device=dev, dtype=torch.long)
        r = torch.rand(B, N, device=dev)

        # If we heard about food, sometimes go there.
        go = heard_food & (r < self.p_teleport)
        act = torch.where(go, torch.full_like(act, TELEPORT), act)

        # If not on food, sometimes listen for a tip.
        listen = (~on_food) & (r >= self.p_teleport) & (r < self.p_teleport + self.p_listen)
        act = torch.where(listen, torch.full_like(act, LISTEN), act)

        # On food: mostly eat, sometimes broadcast the location.
        broadcast = on_food & (r < self.p_broadcast)
        act = torch.where(on_food, torch.full_like(act, EAT), act)
        act = torch.where(broadcast, torch.full_like(act, BROADCAST), act)
        return act


POLICIES = {p.name: p for p in [RandomPolicy, HeuristicPolicy]}


def make_policy(name: str):
    cls = POLICIES.get(name, HeuristicPolicy)
    return cls()
