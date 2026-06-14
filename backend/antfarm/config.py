"""
Simulation configuration for the Communicating Ants project.

Everything that used to be a slider in the 1997 MATLAB GUI lives here as a
single, typed config object. The web frontend edits these values live; the
env reads them. Keeping them in one dataclass makes the "knobs" explicit and
keeps the env code free of magic numbers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import torch


def pick_device(prefer: str = "auto") -> str:
    """Choose a torch device.

    On this M2 Ultra the interesting option is "mps" (Apple's Metal GPU
    backend). There is no CUDA on a Mac. We fall back to CPU if MPS is
    unavailable so the code still runs anywhere.
    """
    if prefer != "auto":
        return prefer
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class SimConfig:
    # --- the world -------------------------------------------------------
    world_size: int = 48          # side length W of the toroidal grid
    n_ants: int = 120             # ants per world
    n_worlds: int = 1             # parallel worlds (batch dim). 1 for Lesson 0.

    # --- energy / life ---------------------------------------------------
    energy_max: float = 20.0      # full tank (was NEnergyStates)
    energy_cost: float = 0.30     # energy burned per step, every ant (was ENCOST)

    # --- food ------------------------------------------------------------
    food_density: float = 0.04    # target fraction of cells holding food
    max_food_size: float = 60.0   # max amount at a freshly spawned food cell
    bite_size: float = 1.0        # most one ant can eat per step (was BASEFOOD)

    # --- movement / communication ---------------------------------------
    move_radius: float = 2.5      # how far a random move can travel
    comm_radius: float = 8.0      # broadcast/listen range (was MASKSIZE)

    # --- ecosystem / population dynamics (Lesson 0.5) -------------------
    # When ecosystem=False we are in the Lesson-0 "homeostatic" world: food
    # refills to a target density and dead ants instantly respawn (population
    # is pinned). When True the world becomes a coupled consumer-resource
    # system: food grows logistically and the population is FREE to boom, bust,
    # oscillate, or go extinct.
    ecosystem: bool = False
    max_ants: int = 1200          # slot capacity for births (tensor size in eco mode)
    food_growth_rate: float = 1.9  # r -- logistic regrowth rate; THE bifurcation knob
    food_diffusion: float = 0.08   # D -- how fast food spreads to neighbour cells
    food_seed: float = 0.02        # spore-rain so grazed-out / extinct food can recover
    birth_threshold: float = 0.80  # fraction of energy_max needed to reproduce
    birth_cost: float = 0.45       # fraction of energy_max handed to each offspring
    enable_comm: bool = True       # turn off the O(N^2) comm step (e.g. for sweeps)

    # --- simulation control ---------------------------------------------
    seed: int = 0
    device: str = "auto"          # "auto" -> mps if present else cpu

    # --- validation ------------------------------------------------------
    def __post_init__(self):
        """Clamp every field to a sane range so no client input (REST query, or
        a config/reset message over the WebSocket) can allocate an absurd tensor,
        divide by zero, or push the food PDE past its stability limit. Runs on
        every construction, including `from_dict` and `with_updates`."""
        def clamp(v, lo, hi):
            return max(lo, min(hi, v))

        # structural sizes -- the dangerous ones for memory
        self.world_size = int(clamp(int(self.world_size), 2, 1024))
        self.n_worlds = int(clamp(int(self.n_worlds), 1, 4096))
        self.n_ants = int(clamp(int(self.n_ants), 1, 200_000))
        self.max_ants = int(clamp(int(self.max_ants), self.n_ants, 200_000))

        # energy / food economy -- non-negative, finite
        self.energy_max = float(clamp(self.energy_max, 1.0, 1e6))
        self.energy_cost = float(clamp(self.energy_cost, 0.0, 1e6))
        self.max_food_size = float(clamp(self.max_food_size, 1e-3, 1e6))
        self.bite_size = float(clamp(self.bite_size, 0.0, 1e6))
        self.food_density = float(clamp(self.food_density, 0.0, 1.0))
        self.move_radius = float(clamp(self.move_radius, 0.0, 1e4))
        self.comm_radius = float(clamp(self.comm_radius, 0.0, 1e4))

        # ecosystem dynamics
        self.food_growth_rate = float(clamp(self.food_growth_rate, 0.0, 4.0))
        # explicit 4-neighbour diffusion is stable only up to ~0.25
        self.food_diffusion = float(clamp(self.food_diffusion, 0.0, 0.25))
        self.food_seed = float(clamp(self.food_seed, 0.0, 1.0))
        self.birth_threshold = float(clamp(self.birth_threshold, 0.0, 1.0))
        self.birth_cost = float(clamp(self.birth_cost, 0.0, 1.0))
        self.seed = int(self.seed)

    # --- bookkeeping -----------------------------------------------------
    def resolved_device(self) -> str:
        return pick_device(self.device)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> SimConfig:
        # Only accept keys we know about; ignore anything stray from the client.
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        clean = {k: v for k, v in d.items() if k in known}
        return cls(**clean)

    def with_updates(self, **kw) -> SimConfig:
        return replace(self, **kw)
