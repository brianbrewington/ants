"""
Presentation: turn an AntWorld into a JSON-ready picture for the browser.

This lives OUTSIDE the simulation core so the env stays headless — a training
run never needs to serialize world-0 ants, build food lists, or carry viz state.
`AntWorld.snapshot()` is kept as a thin delegate to this so existing callers (the
server, tests) don't change.
"""

from __future__ import annotations

import torch

from .contracts import Snapshot


def to_snapshot(env) -> Snapshot:
    """JSON-ready picture of world 0 — alive ants only."""
    cfg = env.cfg
    alive0 = env.alive[0].detach().to("cpu")
    idx = torch.nonzero(alive0, as_tuple=False).flatten()
    pos0 = env.pos[0].detach().to("cpu")[idx]
    en0 = env.energy[0].detach().to("cpu")[idx]
    act0 = env.last_actions[0].detach().to("cpu")[idx]

    food0 = env.food[0].detach().to("cpu")
    nz = torch.nonzero(food0 > 0, as_tuple=False)
    food_list = [[int(x), int(y), float(food0[x, y])] for x, y in nz.tolist()]

    return {
        "world_size": cfg.world_size,
        "ants": {"pos": pos0.tolist(), "energy": en0.tolist(), "action": act0.tolist()},
        "food": food_list,
        "links": env.last_info.get("links", []),
        "metrics": {k: v for k, v in env.last_info.items() if k != "links"},
        "energy_max": cfg.energy_max,
        "max_food": cfg.max_food_size,
        "ecosystem": cfg.ecosystem,
    }
