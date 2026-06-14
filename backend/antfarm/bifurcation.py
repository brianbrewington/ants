"""
Bifurcation sweep.

The batched architecture, cashed in: run many worlds at once, each with a
*different* food growth rate r, settle past the transient, then record the
population over a sampling window. Plotting those samples against r is a
bifurcation diagram -- the period-doubling / boom-bust route, emergent from ants
foraging a renewable resource.

`runs_per_r` adds REPLICATES: each r is simulated in several independent worlds
(same r, different random realization). Because the world is stochastic, a single
run near the extinction threshold can die or survive by luck; replicates separate
a real island of stability from a lucky draw. (This is genuinely different from
sampling r vs r+eps: replicates hold r fixed and vary only the noise.)

Reading it:
  * low r   -> food can't keep up -> population collapses to ~0 (extinction).
  * mid r   -> a single stable level (dots collapse to a thin line).
  * onset & high r -> the line widens/splits: boom-bust oscillation toward chaos.

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
                      runs_per_r: int = 1, max_ants: int = 3000,
                      per_r_points: int = 80) -> dict:
    n_r = max(2, n_r)
    runs_per_r = max(1, runs_per_r)
    total = n_r * runs_per_r  # one world per (r, replicate)

    cfg = SimConfig.from_dict({
        **base,
        "ecosystem": True,
        "n_worlds": total,
        "max_ants": max_ants,   # high enough that FOOD, not the slot cap, limits the pop
        "enable_comm": False,   # isolate population<->food dynamics; skip O(N^2) comm
    })
    env = AntWorld(cfg)
    r_unique = torch.linspace(r_min, r_max, n_r, device=env.device)
    # world w = r_index*runs_per_r + replicate  -> repeat each r `runs_per_r` times.
    env.growth_rate_vec = r_unique.repeat_interleave(runs_per_r)

    for _ in range(transient):
        env.step(ForagePolicy()(env), light=True)

    series = []
    for _ in range(sample):
        env.step(ForagePolicy()(env), light=True)
        series.append(env.alive.sum(dim=1))               # population per world [total]
    pops = torch.stack(series, dim=0).float()              # [sample, total]
    pops = pops.view(sample, n_r, runs_per_r)              # split out replicates

    ref = float(pops.max().item()) or 1.0                  # scale so the diagram fills the plot
    # Aggregate each r across BOTH time and replicates.
    per_r = (pops / ref).permute(1, 0, 2).reshape(n_r, sample * runs_per_r)

    r_list = r_unique.tolist()
    stride = max(1, (sample * runs_per_r) // per_r_points)
    sub = per_r[:, ::stride].tolist()                      # [n_r][~per_r_points]
    points = [[r_list[i], v] for i, vals in enumerate(sub) for v in vals]

    return {
        "r": r_list,
        "points": points,                                  # scaled 0..1
        "min": per_r.min(dim=1).values.tolist(),
        "max": per_r.max(dim=1).values.tolist(),
        "mean": per_r.mean(dim=1).tolist(),
        "pop_ref": ref,
        "runs_per_r": runs_per_r,
        "n_slots": env.n_slots,
    }
