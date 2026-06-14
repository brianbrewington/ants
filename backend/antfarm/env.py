"""
The Communicating Ants world, rebuilt as a *vectorized* simulation.

KEY IDEA (and the whole reason this is fast on the M2 Ultra GPU):
the 1997 MATLAB code looped over ants one at a time. We never loop over ants.
Every ant attribute is a tensor with a batch layout of

        [n_worlds, n_ants]            (or [n_worlds, n_ants, 2] for positions)

so "step every ant in every parallel world" is a handful of tensor ops that
the GPU runs in parallel. `n_worlds` is the batch dimension we'll later use to
run hundreds of simulations at once for reinforcement learning ("parallel
rollouts"). For Lesson 0 we run a single world and just watch it.

The world is a torus: walking off the right edge brings you back on the left.
Positions are continuous floats; the grid cell an ant occupies is floor(pos).

Actions (one integer per ant per step):
    0 EAT        eat food in the current cell (if any, and not full)
    1 BROADCAST  announce your location to nearby listeners
    2 NOTHING    rest
    3 TELEPORT   jump to your stored destination (set by a previous LISTEN)
    4 LISTEN     adopt the location of the nearest broadcaster as your dest
    5 RANDMOVE   wander a short random step

These are exactly the six actions from the original project. In later lessons
a learned policy will choose them; here a simple policy does.
"""

from __future__ import annotations

import torch

from .config import SimConfig

# Action ids -- imported by policies and the server.
EAT, BROADCAST, NOTHING, TELEPORT, LISTEN, RANDMOVE = range(6)
ACTION_NAMES = ["eat", "broadcast", "nothing", "teleport", "listen", "randmove"]
N_ACTIONS = 6


class AntWorld:
    def __init__(self, config: SimConfig):
        self.cfg = config
        self.device = torch.device(config.resolved_device())
        self.step_count = 0
        self.last_info: dict = {}
        self.reset()

    # ------------------------------------------------------------------ #
    # Allocation / reset
    # ------------------------------------------------------------------ #
    def reset(self):
        cfg = self.cfg
        B, N, W = cfg.n_worlds, cfg.n_ants, cfg.world_size
        g = torch.Generator(device="cpu").manual_seed(cfg.seed)
        self._gen = g  # kept for reproducible respawns/food on CPU paths

        dev = self.device
        f32 = torch.float32

        # Ant state -----------------------------------------------------
        # Continuous positions in [0, W).
        self.pos = torch.rand(B, N, 2, generator=g).to(dev) * W
        self.energy = torch.full((B, N), cfg.energy_max, device=dev, dtype=f32)
        # Destination an ant will TELEPORT to (set when it LISTENs).
        self.dest = torch.rand(B, N, 2, generator=g).to(dev) * W
        # Was the last signal we heard coming from an ant that was on food?
        self.heard_food = torch.zeros(B, N, device=dev, dtype=f32)
        # Remember last actions purely so the viz can colour ants by action.
        self.last_actions = torch.full((B, N), NOTHING, device=dev, dtype=torch.long)

        # World state ---------------------------------------------------
        self.food = torch.zeros(B, W, W, device=dev, dtype=f32)
        self.step_count = 0
        self._spawn_food(self._target_food_cells(), only_if_under_target=False)
        self.last_info = self._empty_info()

    def _target_food_cells(self) -> int:
        W = self.cfg.world_size
        return max(1, int(round(self.cfg.food_density * W * W)))

    # ------------------------------------------------------------------ #
    # Food regrowth
    # ------------------------------------------------------------------ #
    def _spawn_food(self, n_per_world: int, only_if_under_target: bool = True):
        """Drop `n_per_world` fresh food piles at random cells in each world.

        Vectorized across worlds. If `only_if_under_target`, worlds that already
        hold enough food are skipped, so the food supply hovers near the target
        density instead of growing without bound.
        """
        if n_per_world <= 0:
            return
        cfg = self.cfg
        B, W = cfg.n_worlds, cfg.world_size
        dev = self.device
        K = n_per_world

        b_idx = torch.arange(B, device=dev).repeat_interleave(K)
        xs = torch.randint(0, W, (B * K,), device=dev)
        ys = torch.randint(0, W, (B * K,), device=dev)
        amounts = (0.2 + 0.8 * torch.rand(B * K, device=dev)) * cfg.max_food_size

        if only_if_under_target:
            count = (self.food > 0).flatten(1).sum(dim=1)          # [B]
            under = (count < self._target_food_cells()).float()     # [B]
            amounts = amounts * under[b_idx]

        self.food.index_put_((b_idx, xs, ys), amounts, accumulate=True)

    # ------------------------------------------------------------------ #
    # Observation -- what a policy is allowed to "see" this step.
    # (In RL lessons this becomes the agent's input. Here a simple policy
    #  uses it. Keeping it in one place keeps env and brain cleanly separated.)
    # ------------------------------------------------------------------ #
    def _cells(self) -> tuple[torch.Tensor, torch.Tensor]:
        W = self.cfg.world_size
        cx = self.pos[..., 0].long().clamp_(0, W - 1)
        cy = self.pos[..., 1].long().clamp_(0, W - 1)
        return cx, cy

    def _food_at_ant(self) -> torch.Tensor:
        cx, cy = self._cells()
        b = torch.arange(self.cfg.n_worlds, device=self.device)[:, None]
        return self.food[b, cx, cy]  # [B, N]

    def observe(self) -> dict:
        on_food = (self._food_at_ant() > 0).float()
        return {
            "on_food": on_food,
            "energy": self.energy,
            "heard_food": self.heard_food,
        }

    # ------------------------------------------------------------------ #
    # The step: apply one action per ant in every world.
    # ------------------------------------------------------------------ #
    def step(self, actions: torch.Tensor) -> dict:
        cfg = self.cfg
        B, N, W = cfg.n_worlds, cfg.n_ants, cfg.world_size
        dev = self.device
        actions = actions.to(dev).long()
        self.last_actions = actions

        # --- 1. EAT ----------------------------------------------------
        # Several ants can share a cell, so we can't let each take a full bite
        # blindly -- they'd eat more food than exists. We compute total *demand*
        # per cell, compare to what's available, and scale every ant's bite by
        # the same fraction. This conserves food exactly.
        cx, cy = self._cells()
        flat_cell = (torch.arange(B, device=dev)[:, None] * (W * W) + cx * W + cy)  # [B,N]
        room = (cfg.energy_max - self.energy).clamp(min=0.0)
        want = torch.minimum(torch.full_like(room, cfg.bite_size), room)
        is_eat = (actions == EAT).float()
        desired = is_eat * want                                    # [B,N]

        food_flat = self.food.view(-1)                             # [B*W*W]
        fc = flat_cell.reshape(-1)
        demand = torch.zeros_like(food_flat).scatter_add_(0, fc, desired.reshape(-1))
        avail_at = food_flat.gather(0, fc)
        demand_at = demand.gather(0, fc)
        ratio = torch.where(demand_at > 0, (avail_at / demand_at).clamp(max=1.0),
                            torch.zeros_like(demand_at))
        bite = desired.reshape(-1) * ratio
        self.energy = self.energy + bite.view(B, N)
        food_flat.scatter_add_(0, fc, -bite)
        self.food.clamp_(min=0.0)
        food_eaten = float(bite.sum().item())

        # --- 2. MOVE (random wander) and TELEPORT ----------------------
        is_move = (actions == RANDMOVE)
        ang = torch.rand(B, N, device=dev) * (2 * torch.pi)
        rad = torch.rand(B, N, device=dev) * cfg.move_radius
        step_vec = torch.stack([torch.cos(ang) * rad, torch.sin(ang) * rad], dim=-1)
        self.pos = torch.where(is_move[..., None], self.pos + step_vec, self.pos)

        is_tp = (actions == TELEPORT)
        self.pos = torch.where(is_tp[..., None], self.dest, self.pos)

        # Toroidal wrap: keep positions in [0, W).
        self.pos = self.pos.remainder(W)

        # --- 3. COMMUNICATE (broadcast / listen) -----------------------
        on_food = (self._food_at_ant() > 0)
        links = self._communicate(actions, on_food)

        # --- 4. METABOLISM: every ant burns energy each step -----------
        self.energy = self.energy - cfg.energy_cost

        # --- 5. DEATH & RESPAWN ----------------------------------------
        dead = self.energy <= 0
        n_dead = int(dead.sum().item())
        if n_dead > 0:
            self._respawn(dead)

        # --- 6. FOOD REGROWTH ------------------------------------------
        self._spawn_food(max(1, self._target_food_cells() // 8))

        self.step_count += 1

        # --- 7. METRICS for the dashboard ------------------------------
        info = {
            "step": self.step_count,
            "food_eaten": food_eaten,
            "deaths": n_dead,
            "mean_energy": float(self.energy.mean().item()),
            "total_food": float(self.food.sum(dim=(1, 2)).mean().item()),
            "frac_eat": float((actions == EAT).float().mean().item()),
            "frac_broadcast": float((actions == BROADCAST).float().mean().item()),
            "frac_listen": float((actions == LISTEN).float().mean().item()),
            "frac_move": float((actions == RANDMOVE).float().mean().item()),
            "links": links,  # world-0 comm links for the viz
        }
        self.last_info = info
        return info

    def _communicate(self, actions: torch.Tensor, on_food: torch.Tensor):
        """Each LISTENer locks onto the nearest in-range BROADCASTer.

        Returns the listener->broadcaster links *in world 0* for visualization.
        Computed with an all-pairs distance tensor -- simple and fully on-GPU.
        For very large ant counts we'd switch to spatial binning; at these sizes
        the [worlds, ants, ants] tensor is tiny.
        """
        cfg = self.cfg
        B, N, W = cfg.n_worlds, cfg.n_ants, cfg.world_size
        dev = self.device

        diff = self.pos[:, :, None, :] - self.pos[:, None, :, :]   # [B,N,N,2]
        diff = (diff + W / 2).remainder(W) - W / 2                  # shortest toroidal vector
        dist = diff.norm(dim=-1)                                    # [B,N,N]

        is_broadcaster = (actions == BROADCAST)                     # [B,N]
        eye = torch.eye(N, device=dev, dtype=torch.bool)
        valid = is_broadcaster[:, None, :] & (dist <= cfg.comm_radius) & (~eye)
        big = torch.full_like(dist, float("inf"))
        dist_masked = torch.where(valid, dist, big)                 # [B,N,N]

        nearest = dist_masked.argmin(dim=-1)                        # [B,N]
        nearest_dist = dist_masked.gather(-1, nearest[..., None]).squeeze(-1)
        has_signal = torch.isfinite(nearest_dist)
        is_listener = (actions == LISTEN)
        update = is_listener & has_signal                          # [B,N]

        src_pos = torch.gather(self.pos, 1, nearest[..., None].expand(-1, -1, 2))
        self.dest = torch.where(update[..., None], src_pos, self.dest)
        src_on_food = torch.gather(on_food.float(), 1, nearest)
        self.heard_food = torch.where(update, src_on_food, self.heard_food)

        # Extract world-0 links for the frontend (cheap, runs once).
        links = []
        if B > 0:
            li = torch.nonzero(update[0], as_tuple=False).flatten().tolist()
            for i in li:
                j = int(nearest[0, i].item())
                lx, ly = self.pos[0, i].tolist()
                sx, sy = self.pos[0, j].tolist()
                links.append([[lx, ly], [sx, sy]])
        return links

    def _respawn(self, dead: torch.Tensor):
        """Replace dead ants with fresh, full-energy ones at random spots."""
        cfg = self.cfg
        W = cfg.world_size
        idx = dead.nonzero(as_tuple=False)
        n = idx.shape[0]
        new_pos = torch.rand(n, 2, device=self.device) * W
        self.pos[dead] = new_pos
        self.dest[dead] = torch.rand(n, 2, device=self.device) * W
        self.energy[dead] = cfg.energy_max
        self.heard_food[dead] = 0.0

    # ------------------------------------------------------------------ #
    # Serialization for the web frontend (world 0 only).
    # ------------------------------------------------------------------ #
    def _empty_info(self) -> dict:
        return {"step": 0, "food_eaten": 0.0, "deaths": 0, "mean_energy": 0.0,
                "total_food": 0.0, "frac_eat": 0.0, "frac_broadcast": 0.0,
                "frac_listen": 0.0, "frac_move": 0.0, "links": []}

    def snapshot(self) -> dict:
        """A JSON-ready picture of world 0: ants, food, comm links, metrics."""
        W = self.cfg.world_size
        pos0 = self.pos[0].detach().to("cpu")
        en0 = self.energy[0].detach().to("cpu")
        act0 = self.last_actions[0].detach().to("cpu")

        # nonzero food cells -> [x, y, amount]
        food0 = self.food[0].detach().to("cpu")
        nz = torch.nonzero(food0 > 0, as_tuple=False)
        food_list = [[int(x), int(y), float(food0[x, y])] for x, y in nz.tolist()]

        return {
            "world_size": W,
            "ants": {
                "pos": pos0.tolist(),
                "energy": en0.tolist(),
                "action": act0.tolist(),
            },
            "food": food_list,
            "links": self.last_info.get("links", []),
            "metrics": {k: v for k, v in self.last_info.items() if k != "links"},
            "energy_max": self.cfg.energy_max,
        }
