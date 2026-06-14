"""
Typed contracts at the seams — what the env produces and consumes.

These make the boundaries explicit (the architecture review flagged them as
convention-only) and give the RL phases a stable surface to build on. They are
TypedDicts: zero runtime cost, dict-compatible with existing callers, but now
documented and checkable.
"""

from __future__ import annotations

from typing import TypedDict

import torch


class StepInfo(TypedDict):
    """Per-step facts returned by AntWorld.step(). Event counts (deaths/births/
    food_eaten) are for THAT step; the server aggregates them across a frame."""
    step: int
    population: int
    food_eaten: float
    deaths: int
    births: int
    mean_energy: float
    total_food: float
    total_nutrient: float
    frac_eat: float
    frac_broadcast: float
    frac_nothing: float
    frac_teleport: float
    frac_listen: float
    frac_move: float
    frac_reproduce: float
    links: list  # world-0 communication links (viz only)


class Observation(TypedDict):
    """The core per-ant observation, all tensors [n_worlds, n_slots]. The legal-
    action mask is a separate surface (AntWorld.action_mask(), shape
    [n_worlds, n_slots, N_ACTIONS]) so it's only computed when a learner needs it.
    """
    on_food: torch.Tensor
    energy: torch.Tensor
    heard_food: torch.Tensor
    alive: torch.Tensor


class FrameMetrics(StepInfo, total=False):
    """What the server streams per frame: the StepInfo fields, but with
    births/deaths/food_eaten summed and frac_* averaged across the frame's steps,
    and `links` dropped (it's viz, carried in the snapshot)."""


class AntsView(TypedDict):
    pos: list      # [[x, y], ...] alive ants in world 0
    energy: list
    action: list


class Snapshot(TypedDict):
    """World-0 picture sent to the browser (see snapshot.to_snapshot)."""
    world_size: int
    ants: AntsView
    food: list      # [[x, y, amount], ...]
    links: list     # [[[lx,ly],[sx,sy]], ...] communication links
    metrics: dict   # StepInfo (paused) or FrameMetrics (running), set by the server
    energy_max: float
    max_food: float
    ecosystem: bool
