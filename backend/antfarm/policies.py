"""
"Brains" for the ants. Contract:  actions = policy(env)  -> LongTensor [B, n_slots]

A learned policy (Q-table, neural net) will slot in behind this exact interface
in later lessons. For now:

  * RandomPolicy   -- pure noise (motivates the need for a brain).
  * HeuristicPolicy -- a hand-written rule of thumb; eats, sometimes talks,
                       and (in ecosystem mode) reproduces when well-fed.
  * ForagePolicy   -- eat / wander / reproduce only, no communication. Used for
                       the bifurcation sweep so we skip the O(N^2) comm step and
                       isolate the population<->food dynamics.
"""

from __future__ import annotations

import torch

from .env import BROADCAST, EAT, LISTEN, N_ACTIONS, RANDMOVE, REPRODUCE, TELEPORT, AntWorld


class RandomPolicy:
    name = "random"

    def __call__(self, env: AntWorld) -> torch.Tensor:
        return torch.randint(0, N_ACTIONS, (env.cfg.n_worlds, env.n_slots),
                             device=env.device, generator=env.gen)


class HeuristicPolicy:
    name = "heuristic"

    def __init__(self, p_broadcast=0.15, p_listen=0.25, p_teleport=0.5, p_reproduce=0.3):
        self.p_broadcast = p_broadcast
        self.p_listen = p_listen
        self.p_teleport = p_teleport
        self.p_reproduce = p_reproduce

    def __call__(self, env: AntWorld) -> torch.Tensor:
        obs = env.observe()
        B, S, dev = env.cfg.n_worlds, env.n_slots, env.device
        on_food = obs["on_food"].bool()
        heard_food = obs["heard_food"].bool()
        energy = obs["energy"]

        act = torch.full((B, S), RANDMOVE, device=dev, dtype=torch.long)
        r = torch.rand(B, S, device=dev, generator=env.gen)

        # heard about food -> sometimes go there
        act = torch.where(heard_food & (r < self.p_teleport),
                          torch.full_like(act, TELEPORT), act)
        # not on food -> sometimes listen
        listen = (~on_food) & (r >= self.p_teleport) & (r < self.p_teleport + self.p_listen)
        act = torch.where(listen, torch.full_like(act, LISTEN), act)
        # on food -> eat, occasionally broadcast
        act = torch.where(on_food, torch.full_like(act, EAT), act)
        act = torch.where(on_food & (r < self.p_broadcast),
                          torch.full_like(act, BROADCAST), act)

        # well-fed -> reproduce (overrides eating; a full ant gains little from
        # more food). birth_threshold should sit comfortably above birth_cost so
        # the parent survives the split. Only meaningful in ecosystem modes.
        if env.cfg.ecosystem:
            full = energy >= env.cfg.birth_threshold * env.cfg.energy_max
            act = torch.where(full & (r < self.p_reproduce),
                              torch.full_like(act, REPRODUCE), act)
        return act


class ForagePolicy:
    """No communication. Eat when on food, reproduce when full, else wander."""
    name = "forage"

    def __init__(self, p_reproduce=0.3):
        self.p_reproduce = p_reproduce

    def __call__(self, env: AntWorld) -> torch.Tensor:
        obs = env.observe()
        B, S, dev = env.cfg.n_worlds, env.n_slots, env.device
        on_food = obs["on_food"].bool()
        energy = obs["energy"]

        act = torch.full((B, S), RANDMOVE, device=dev, dtype=torch.long)
        act = torch.where(on_food, torch.full_like(act, EAT), act)
        if env.cfg.ecosystem:
            r = torch.rand(B, S, device=dev, generator=env.gen)
            full = energy >= env.cfg.birth_threshold * env.cfg.energy_max
            act = torch.where(full & (r < self.p_reproduce),
                              torch.full_like(act, REPRODUCE), act)
        return act


POLICIES = {p.name: p for p in [RandomPolicy, HeuristicPolicy, ForagePolicy]}


def make_policy(name: str):
    # REVIEW-NOTE 2026-06-14 (deferred): unknown names silently fall back to the
    # heuristic. Acceptable because the frontend only sends valid names; if we add
    # a public API we should reject unknown policies instead. See
    # docs/review-responses/2026-06-14-gpt-5.5.md.
    return POLICIES.get(name, HeuristicPolicy)()
