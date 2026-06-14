"""
Simulation configuration for the Communicating Ants project.

Everything that used to be a slider in the 1997 MATLAB GUI lives here as a
single, typed config object. The web frontend edits these values live; the
env reads them. Keeping them in one dataclass makes the "knobs" explicit and
keeps the env code free of magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
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

    # --- simulation control ---------------------------------------------
    seed: int = 0
    device: str = "auto"          # "auto" -> mps if present else cpu

    # --- bookkeeping -----------------------------------------------------
    def resolved_device(self) -> str:
        return pick_device(self.device)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SimConfig":
        # Only accept keys we know about; ignore anything stray from the client.
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        clean = {k: v for k, v in d.items() if k in known}
        return cls(**clean)

    def with_updates(self, **kw) -> "SimConfig":
        return replace(self, **kw)
