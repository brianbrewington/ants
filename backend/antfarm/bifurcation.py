"""
Bifurcation sweep.

The batched architecture, cashed in: run many worlds at once, each with a
*different* food growth rate r, settle past the transient, then record the
population over a sampling window. Plotting those samples against r is a
bifurcation diagram -- the period-doubling / boom-bust route, emergent from ants
foraging a renewable resource.

Reading it:
  * low r   -> food can't keep up -> population collapses to ~0 (extinction).
  * mid r   -> a single stable level (dots collapse to a thin line).
  * onset & high r -> the line widens/splits: boom-bust oscillation (predator-
    prey limit cycle) toward chaos -- the "paradox of enrichment".

Speed: every step uses env.step(light=True), which skips the per-step GPU->CPU
metric syncs. We only pull population off the GPU once, at the end.
"""

from __future__ import annotations

import torch

from .config import SimConfig
from .env import AntWorld
from .policies import ForagePolicy


def bifurcation_sweep(base: dict, r_min: float = 0.3, r_max: float = 3.0,
                      n_r: int = 200, transient: int = 400, sample: int = 240,
                      max_ants: int = 6000, per_r_points: int = 70) -> dict:
    cfg = SimConfig.from_dict({
        **base,
        "ecosystem": True,
        "n_worlds": n_r,
        "max_ants": max_ants,   # high enough that FOOD, not the slot cap, limits the pop
        "enable_comm": False,   # isolate population<->food dynamics; skip O(N^2) comm
    })
    env = AntWorld(cfg)
    env.growth_rate_vec = torch.linspace(r_min, r_max, n_r, device=env.device)
    policy = ForagePolicy()

    for _ in range(transient):
        env.step(policy(env), light=True)

    series = []
    for _ in range(sample):
        env.step(policy(env), light=True)
        series.append(env.alive.sum(dim=1))           # population per world, stays on GPU
    pops = torch.stack(series, dim=0).float()          # [sample, n_r]  (one sync below)

    r_list = env.growth_rate_vec.tolist()
    pmin = pops.min(dim=0).values
    pmax = pops.max(dim=0).values
    pmean = pops.mean(dim=0)
    ref = float(pmax.max().item()) or 1.0              # scale so the diagram fills the plot

    # Sub-sample the window so the attractor shows without a huge payload.
    stride = max(1, sample // per_r_points)
    sub = (pops[::stride] / ref).t().tolist()          # [n_r][~per_r_points], 0..1
    points = [[r_list[i], v] for i, vals in enumerate(sub) for v in vals]

    return {
        "r": r_list,
        "points": points,                              # already scaled 0..1 by ref
        "min": (pmin / ref).tolist(),
        "max": (pmax / ref).tolist(),
        "mean": (pmean / ref).tolist(),
        "pop_ref": ref,                                # actual ant count at y=1.0
        "n_slots": env.n_slots,
    }
