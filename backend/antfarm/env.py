"""
The Communicating Ants world, vectorized for the GPU.

Two worlds in one, chosen by `config.ecosystem`:

  * HOMEOSTATIC (Lesson 0, ecosystem=False)
      Food refills toward a target density; dead ants instantly respawn so the
      population is pinned at a constant N. Nothing can really collapse -- it's a
      thermostat. Good for a calm substrate to learn on.

  * ECOSYSTEM (Lesson 0.5, ecosystem=True)
      A coupled consumer-resource system. Food grows *logistically* with rate r
      (the bifurcation knob), spreads by diffusion, and is grazed down by ants.
      Ants die for real (no respawn) and REPRODUCE when well-fed. Population is a
      free variable -> it can stabilize, boom-bust (predator-prey limit cycle),
      go chaotic, or go extinct depending on r. See docs/lessons/lesson-0.5.

KEY VECTORIZATION IDEAS
  * Batch layout [n_worlds, n_slots]: we never loop over ants.
  * Variable population stays fixed-size on the GPU via a *slot pool*: we
    allocate n_slots columns and carry a boolean `alive` mask. Births fill dead
    slots; deaths clear the mask. Dead slots are forced to do nothing and are
    filtered out of the view. "No free slot" becomes a natural carrying capacity.

Actions (per ant per step):
    0 EAT  1 BROADCAST  2 NOTHING  3 TELEPORT  4 LISTEN  5 RANDMOVE  6 REPRODUCE
"""

from __future__ import annotations

import torch

from .config import SimConfig

EAT, BROADCAST, NOTHING, TELEPORT, LISTEN, RANDMOVE, REPRODUCE = range(7)
ACTION_NAMES = ["eat", "broadcast", "nothing", "teleport", "listen", "randmove", "reproduce"]
N_ACTIONS = 7

# Communication builds a pairwise [n_worlds, n_slots, n_slots] tensor. Above this
# many pairs we skip it to avoid an abrupt huge allocation (MPS handles those
# badly).
# REVIEW-NOTE 2026-06-14 (deferred): this CAP is a guard, not the real fix. Proper
# spatial binning (only check nearby ants) lands with Lesson 0.6 when populations
# scale up. See docs/review-responses/2026-06-14-gpt-5.5.md.
COMM_PAIR_CAP = 50_000_000


class AntWorld:
    def __init__(self, config: SimConfig):
        self.cfg = config
        self.device = torch.device(config.resolved_device())
        # In ecosystem mode we over-allocate columns so births have room.
        self.n_slots = config.max_ants if config.ecosystem else config.n_ants
        self.n_slots = max(self.n_slots, config.n_ants)
        self.step_count = 0
        self.last_info: dict = {}
        # Optional per-world override of the food growth rate r, shape [n_worlds].
        # Used by the bifurcation sweep so every parallel world runs a different r.
        self.growth_rate_vec: torch.Tensor | None = None
        self.reset()

    # ------------------------------------------------------------------ #
    # Allocation / reset
    # ------------------------------------------------------------------ #
    def reset(self):
        cfg = self.cfg
        B, S, W = cfg.n_worlds, self.n_slots, cfg.world_size
        dev = self.device
        f32 = torch.float32
        # ONE per-env generator drives ALL randomness -- initial placement *and*
        # every per-step dynamic (movement, spores, respawn, births). This is
        # what makes `seed` actually reproducible and isolates each env's RNG
        # stream from the global one (so a sweep can't perturb the live sim).
        g = self.gen = torch.Generator(device=dev).manual_seed(cfg.seed)

        self.pos = torch.rand(B, S, 2, device=dev, generator=g) * W
        self.energy = torch.zeros(B, S, device=dev, dtype=f32)
        self.dest = torch.rand(B, S, 2, device=dev, generator=g) * W
        self.heard_food = torch.zeros(B, S, device=dev, dtype=f32)
        self.last_actions = torch.full((B, S), NOTHING, device=dev, dtype=torch.long)

        # alive mask: first n_ants slots live, the rest are empty (eco) slots.
        self.alive = torch.zeros(B, S, device=dev, dtype=torch.bool)
        self.alive[:, : cfg.n_ants] = True
        # Start everyone with a full tank (Lesson 0). Jitter is left as a knob we
        # can add later; deterministic start makes the homeostatic demo legible.
        self.energy[self.alive] = cfg.energy_max

        # Food (+ nutrient substrate for the closed-loop model) -------
        self.food = torch.zeros(B, W, W, device=dev, dtype=f32)
        self.nutrient = torch.zeros(B, W, W, device=dev, dtype=f32)
        if cfg.ecosystem and cfg.food_model == "nutrient":
            # Most of the world's mass starts in the nutrient pool; food germinates
            # out of it. (Lesson 0.6 — see _grow_food_nutrient.) But seed some
            # initial food patches up front (drawn FROM the nutrient, so mass is
            # still conserved) -- otherwise the starting population starves before
            # germination can build food up from zero.
            self.nutrient = torch.full((B, W, W), cfg.nutrient_init, device=dev, dtype=f32)
            seed = (torch.rand(B, W, W, device=dev, generator=g) < cfg.food_density).float()
            sprout = seed * 0.5 * self.nutrient
            self.food = sprout
            self.nutrient = self.nutrient - sprout
        elif cfg.ecosystem:
            # Sparse food PATCHES, not a carpet (Lesson 0.5): seed only a small
            # fraction of cells; with diffusion off, empty cells stay empty so
            # food stays localized. Spore rain reseeds new patches over time.
            seed = (torch.rand(B, W, W, device=dev, generator=g) < cfg.food_density).float()
            self.food = seed * (0.5 + 0.5 * torch.rand(B, W, W, device=dev, generator=g)) * cfg.max_food_size
        else:
            self._spawn_food(self._target_food_cells(), only_if_under_target=False)

        self.step_count = 0
        self.last_info = self._empty_info()

    def _target_food_cells(self) -> int:
        W = self.cfg.world_size
        return max(1, int(round(self.cfg.food_density * W * W)))

    # ------------------------------------------------------------------ #
    # Food: homeostatic refill (Lesson 0)
    # ------------------------------------------------------------------ #
    def _spawn_food(self, n_per_world: int, only_if_under_target: bool = True):
        if n_per_world <= 0:
            return
        cfg = self.cfg
        B, W = cfg.n_worlds, cfg.world_size
        dev = self.device
        K = n_per_world
        b_idx = torch.arange(B, device=dev).repeat_interleave(K)
        xs = torch.randint(0, W, (B * K,), device=dev, generator=self.gen)
        ys = torch.randint(0, W, (B * K,), device=dev, generator=self.gen)
        amounts = (0.2 + 0.8 * torch.rand(B * K, device=dev, generator=self.gen)) * cfg.max_food_size
        if only_if_under_target:
            count = (self.food > 0).flatten(1).sum(dim=1)
            under = (count < self._target_food_cells()).float()
            amounts = amounts * under[b_idx]
        self.food.index_put_((b_idx, xs, ys), amounts, accumulate=True)

    # ------------------------------------------------------------------ #
    # Food: renewable logistic growth (Lesson 0.5)
    # ------------------------------------------------------------------ #
    def _grow_food(self):
        """Discrete logistic growth + diffusion + a little spore rain.

            F <- F + r*F*(1 - F/K) + D*(neighbour_mean - F) + spores

        * The logistic term is the resource's intrinsic growth -- r is the knob
          that drives the bifurcation cascade (paradox of enrichment).
        * Diffusion lets food spread into empty cells (pure logistic growth from
          F=0 stays 0 forever, so something has to recolonize).
        * Spore rain reseeds rare random cells so a fully grazed-out / extinct
          food field can recover.
        """
        cfg = self.cfg
        K = cfg.max_food_size  # per-cell carrying capacity
        F = self.food

        # logistic growth (r may be a single value or one per world)
        r = cfg.food_growth_rate if self.growth_rate_vec is None \
            else self.growth_rate_vec.view(-1, 1, 1)
        F = F + r * F * (1.0 - F / K)

        # 4-neighbour toroidal diffusion (mass-conserving)
        if cfg.food_diffusion > 0:
            neigh = (torch.roll(F, 1, 1) + torch.roll(F, -1, 1)
                     + torch.roll(F, 1, 2) + torch.roll(F, -1, 2)) * 0.25
            F = F + cfg.food_diffusion * (neigh - F)

        # spore rain: a few random cells get a small seed each step
        if cfg.food_seed > 0:
            B, W = cfg.n_worlds, cfg.world_size
            n = max(1, int(cfg.food_seed * W))  # cells reseeded per world per step
            dev = self.device
            b_idx = torch.arange(B, device=dev).repeat_interleave(n)
            xs = torch.randint(0, W, (B * n,), device=dev, generator=self.gen)
            ys = torch.randint(0, W, (B * n,), device=dev, generator=self.gen)
            F.index_put_((b_idx, xs, ys),
                         torch.full((B * n,), 0.05 * K, device=dev), accumulate=True)

        self.food = F.clamp_(min=0.0, max=4.0 * K)

    # ------------------------------------------------------------------ #
    # Food: closed nutrient loop (Lesson 0.6)
    # ------------------------------------------------------------------ #
    def _deposit_nutrient(self, amount: torch.Tensor):
        """Return per-ant `amount` [n_worlds, n_slots] of mass to the nutrient
        grid at each ant's cell (used by metabolism and death)."""
        B, W = self.cfg.n_worlds, self.cfg.world_size
        cx, cy = self._cells()
        flat = torch.arange(B, device=self.device)[:, None] * (W * W) + cx * W + cy
        self.nutrient.view(-1).scatter_add_(0, flat.reshape(-1), amount.reshape(-1))

    def _grow_food_nutrient(self):
        """Grow food OUT OF the nutrient pool instead of from a hard carrying
        capacity. Two consequences vs the logistic model:

          * No hard cap on a food pile -- growth is throttled by how much nutrient
            is locally left (Monod term N/(N+h)), which falls as food accumulates.
          * A zeroed cell is NOT absorbing: stochastic germination sprouts food
            from the nutrient beneath it (a mass-conserving replacement for spore
            rain). So food can always recolonize while nutrient remains.

        With nutrient_inflow=0 every term just moves mass N<->F, so N+F is
        conserved (and, with metabolism/death returning to N, so is N+F+energy).
        """
        cfg = self.cfg
        B, W = cfg.n_worlds, cfg.world_size
        N, F = self.nutrient, self.food

        # 1. growth on existing food, Monod-limited by local nutrient, drawn from N.
        #    r may be a single value or one-per-world (the bifurcation sweep sets a
        #    per-world vector -- nutrient growth must honor it just like logistic does).
        r = cfg.food_growth_rate if self.growth_rate_vec is None \
            else self.growth_rate_vec.view(-1, 1, 1)
        grow = torch.minimum(r * F * (N / (N + cfg.half_sat)), N)
        F = F + grow
        N = N - grow

        # 2. stochastic germination: a few random cells convert local nutrient into
        #    a fresh food patch (lets F restart from 0; keeps food patchy).
        if cfg.germination > 0:
            n = max(1, int(cfg.germination * W))
            b_idx = torch.arange(B, device=self.device).repeat_interleave(n)
            xs = torch.randint(0, W, (B * n,), device=self.device, generator=self.gen)
            ys = torch.randint(0, W, (B * n,), device=self.device, generator=self.gen)
            sprout = 0.5 * N[b_idx, xs, ys]                 # half the local nutrient germinates
            N.index_put_((b_idx, xs, ys), -sprout, accumulate=True)
            F.index_put_((b_idx, xs, ys), sprout, accumulate=True)

        # 3. nutrient diffusion (mass-conserving), reusing the diffusion knob
        if cfg.food_diffusion > 0:
            neigh = (torch.roll(N, 1, 1) + torch.roll(N, -1, 1)
                     + torch.roll(N, 1, 2) + torch.roll(N, -1, 2)) * 0.25
            N = N + cfg.food_diffusion * (neigh - N)

        # 4. external nutrient inflow ("sunlight"); >0 makes the system OPEN
        if cfg.nutrient_inflow > 0:
            N = N + cfg.nutrient_inflow

        self.nutrient = N.clamp_(min=0.0)
        self.food = F.clamp_(min=0.0)

    # ------------------------------------------------------------------ #
    # Observation (the future RL input). Computed over all slots; the policy
    # may produce actions for dead slots, but step() forces them to NOTHING.
    # ------------------------------------------------------------------ #
    def _cells(self):
        W = self.cfg.world_size
        cx = self.pos[..., 0].long().clamp_(0, W - 1)
        cy = self.pos[..., 1].long().clamp_(0, W - 1)
        return cx, cy

    def _food_at_ant(self):
        cx, cy = self._cells()
        b = torch.arange(self.cfg.n_worlds, device=self.device)[:, None]
        return self.food[b, cx, cy]

    def observe(self) -> dict:
        return {
            "on_food": (self._food_at_ant() > 0).float(),
            "energy": self.energy,
            "heard_food": self.heard_food,
            "alive": self.alive,
        }

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def step(self, actions: torch.Tensor, light: bool = False) -> dict:
        """Advance one tick. `light=True` skips all metric/CPU syncs (used by the
        bifurcation sweep where we only read population, occasionally) -- this is
        the difference between a sluggish sweep and a fast one on MPS."""
        cfg = self.cfg
        B, S, W = cfg.n_worlds, self.n_slots, cfg.world_size
        dev = self.device
        actions = actions.to(dev).long()
        # dead slots never act.
        actions = torch.where(self.alive, actions, torch.full_like(actions, NOTHING))
        self.last_actions = actions

        # --- EAT (contention-safe; conserves food) ---------------------
        cx, cy = self._cells()
        flat_cell = (torch.arange(B, device=dev)[:, None] * (W * W) + cx * W + cy)
        room = (cfg.energy_max - self.energy).clamp(min=0.0)
        want = torch.minimum(torch.full_like(room, cfg.bite_size), room)
        desired = (actions == EAT).float() * want
        food_flat = self.food.view(-1)
        fc = flat_cell.reshape(-1)
        demand = torch.zeros_like(food_flat).scatter_add_(0, fc, desired.reshape(-1))
        avail_at = food_flat.gather(0, fc)
        demand_at = demand.gather(0, fc)
        ratio = torch.where(demand_at > 0, (avail_at / demand_at).clamp(max=1.0),
                            torch.zeros_like(demand_at))
        bite = desired.reshape(-1) * ratio
        self.energy = self.energy + bite.view(B, S)
        food_flat.scatter_add_(0, fc, -bite)
        self.food.clamp_(min=0.0)

        # --- MOVE / TELEPORT ------------------------------------------
        is_move = (actions == RANDMOVE)
        ang = torch.rand(B, S, device=dev, generator=self.gen) * (2 * torch.pi)
        rad = torch.rand(B, S, device=dev, generator=self.gen) * cfg.move_radius
        step_vec = torch.stack([torch.cos(ang) * rad, torch.sin(ang) * rad], dim=-1)
        self.pos = torch.where(is_move[..., None], self.pos + step_vec, self.pos)
        is_tp = (actions == TELEPORT)
        self.pos = torch.where(is_tp[..., None], self.dest, self.pos)
        self.pos = self.pos.remainder(W)

        # --- COMMUNICATE ----------------------------------------------
        on_food = (self._food_at_ant() > 0)
        links = self._communicate(actions, on_food)

        nutrient_mode = cfg.ecosystem and cfg.food_model == "nutrient"

        # --- METABOLISM (alive only) ----------------------------------
        if nutrient_mode:
            # Burn at most what the ant has, and return that mass to the nutrient
            # pool at its cell (closed loop -> mass conserved).
            avail = self.energy.clamp(min=0.0)
            burn = torch.where(self.alive,
                               torch.minimum(torch.full_like(avail, cfg.energy_cost), avail),
                               torch.zeros_like(avail))
            self._deposit_nutrient(burn)
            self.energy = self.energy - burn
        else:
            self.energy = torch.where(self.alive, self.energy - cfg.energy_cost, self.energy)

        # --- BIRTHS & DEATHS ------------------------------------------
        if cfg.ecosystem:
            n_born = self._reproduce(actions, light=light)
            dead = self.alive & (self.energy <= 1e-6)
            if nutrient_mode:
                # any body mass left at death returns to the soil (carcass)
                self._deposit_nutrient(torch.where(dead, self.energy.clamp(min=0.0),
                                                   torch.zeros_like(self.energy)))
            self.alive[dead] = False           # boolean-mask assign: no CPU sync
            self.energy[dead] = 0.0
            if nutrient_mode:
                self._grow_food_nutrient()
            else:
                self._grow_food()
            n_dead = 0 if light else int(dead.sum().item())
        else:
            n_born = 0
            dead = self.energy <= 0
            n_dead = int(dead.sum().item())
            if n_dead:
                self._respawn(dead)
            self._spawn_food(max(1, self._target_food_cells() // 8))

        self.step_count += 1

        # Light mode: skip every metric (each .item() forces a GPU->CPU sync).
        if light:
            return {}

        food_eaten = float(bite.sum().item())
        population = int(self.alive.sum().item())
        info = {
            "step": self.step_count,
            "population": population,
            "food_eaten": food_eaten,
            "deaths": n_dead,
            "births": n_born,
            "mean_energy": float(self.energy[self.alive].mean().item()) if population else 0.0,
            "total_food": float(self.food.sum(dim=(1, 2)).mean().item()),
            "total_nutrient": float(self.nutrient.sum(dim=(1, 2)).mean().item()),
            "frac_eat": self._frac(actions, EAT),
            "frac_broadcast": self._frac(actions, BROADCAST),
            "frac_nothing": self._frac(actions, NOTHING),
            "frac_teleport": self._frac(actions, TELEPORT),
            "frac_listen": self._frac(actions, LISTEN),
            "frac_move": self._frac(actions, RANDMOVE),
            "frac_reproduce": self._frac(actions, REPRODUCE),
            "links": links,
        }
        self.last_info = info
        return info

    def _frac(self, actions, a):
        pop = self.alive.sum()
        if pop == 0:
            return 0.0
        return float(((actions == a) & self.alive).sum().item() / pop.item())

    # ------------------------------------------------------------------ #
    def _communicate(self, actions, on_food):
        cfg = self.cfg
        S, W = self.n_slots, cfg.world_size
        dev = self.device
        # Guard the O(N^2) pairwise tensor against exploding at large populations
        # (cheap check, no GPU sync) before doing any work.
        if not cfg.enable_comm or cfg.n_worlds * self.n_slots ** 2 > COMM_PAIR_CAP:
            return []
        is_broadcaster = (actions == BROADCAST) & self.alive
        is_listener = (actions == LISTEN) & self.alive
        # Skip the rest when nobody is talking/listening.
        if not (bool(is_broadcaster.any()) and bool(is_listener.any())):
            return []

        diff = self.pos[:, :, None, :] - self.pos[:, None, :, :]
        diff = (diff + W / 2).remainder(W) - W / 2
        dist = diff.norm(dim=-1)
        eye = torch.eye(S, device=dev, dtype=torch.bool)
        valid = is_broadcaster[:, None, :] & (dist <= cfg.comm_radius) & (~eye)
        dist_masked = torch.where(valid, dist, torch.full_like(dist, float("inf")))
        nearest = dist_masked.argmin(dim=-1)
        has_signal = torch.isfinite(dist_masked.gather(-1, nearest[..., None]).squeeze(-1))
        update = is_listener & has_signal

        src_pos = torch.gather(self.pos, 1, nearest[..., None].expand(-1, -1, 2))
        self.dest = torch.where(update[..., None], src_pos, self.dest)
        src_on_food = torch.gather(on_food.float(), 1, nearest)
        self.heard_food = torch.where(update, src_on_food, self.heard_food)

        links = []
        for i in torch.nonzero(update[0], as_tuple=False).flatten().tolist():
            j = int(nearest[0, i].item())
            links.append([self.pos[0, i].tolist(), self.pos[0, j].tolist()])
        return links

    def _reproduce(self, actions, light: bool = False) -> int:
        """Well-fed ants that chose REPRODUCE place an offspring in a free slot.

        Fully vectorized slot assignment: rank candidates and free slots within
        each world (cumsum), then map candidate k -> the k-th free slot. No
        Python loop over worlds.
        """
        cfg = self.cfg
        B, S = cfg.n_worlds, self.n_slots
        dev = self.device
        thresh = cfg.birth_threshold * cfg.energy_max
        cost = cfg.birth_cost * cfg.energy_max

        repro = self.alive & (actions == REPRODUCE) & (self.energy >= thresh)
        if not light and not bool(repro.any()):
            return 0
        free = ~self.alive

        cand_rank = torch.cumsum(repro.long(), dim=1) - 1     # 0..(#cand-1) on candidates
        free_rank = torch.cumsum(free.long(), dim=1) - 1      # 0..(#free-1) on free slots
        n_free = free.sum(dim=1)                              # [B]

        jidx = torch.arange(S, device=dev).expand(B, S)
        # slot_for_rank[b, r] = column index of the r-th free slot (dump non-free at col S)
        free_rank_safe = torch.where(free, free_rank, torch.full_like(free_rank, S))
        slot_for_rank = torch.full((B, S + 1), -1, dtype=torch.long, device=dev)
        slot_for_rank.scatter_(1, free_rank_safe, jidx)

        can_place = repro & (cand_rank < n_free[:, None])
        target_rank = torch.where(can_place, cand_rank, torch.full_like(cand_rank, S))
        target_slot = slot_for_rank.gather(1, target_rank.clamp(max=S))

        bsel, isel = can_place.nonzero(as_tuple=True)         # parents
        if bsel.numel() == 0:
            return 0
        tslot = target_slot[bsel, isel]                       # child slots
        # write children
        self.alive[bsel, tslot] = True
        jitter = (torch.rand(bsel.numel(), 2, device=dev, generator=self.gen) - 0.5) * 2.0
        self.pos[bsel, tslot] = (self.pos[bsel, isel] + jitter).remainder(cfg.world_size)
        self.dest[bsel, tslot] = self.pos[bsel, tslot]
        self.energy[bsel, tslot] = cost
        self.heard_food[bsel, tslot] = 0.0
        self.last_actions[bsel, tslot] = NOTHING
        # parents pay
        self.energy[bsel, isel] = self.energy[bsel, isel] - cost
        return int(bsel.numel())

    def _respawn(self, dead):
        cfg = self.cfg
        W = cfg.world_size
        n = int(dead.sum().item())
        self.pos[dead] = torch.rand(n, 2, device=self.device, generator=self.gen) * W
        self.dest[dead] = torch.rand(n, 2, device=self.device, generator=self.gen) * W
        self.energy[dead] = cfg.energy_max
        self.heard_food[dead] = 0.0
        self.alive[dead] = True

    # ------------------------------------------------------------------ #
    def _empty_info(self) -> dict:
        # Describe the freshly-reset world truthfully (ants start alive with energy;
        # food may already be present), so the very first frame's metrics match it.
        pop = int(self.alive.sum().item())
        mean_e = float(self.energy[self.alive].mean().item()) if pop else 0.0
        total_food = float(self.food.sum(dim=(1, 2)).mean().item())
        total_nutrient = float(self.nutrient.sum(dim=(1, 2)).mean().item())
        return {"step": 0, "population": pop, "food_eaten": 0.0,
                "deaths": 0, "births": 0, "mean_energy": mean_e, "total_food": total_food,
                "total_nutrient": total_nutrient,
                "frac_eat": 0.0, "frac_broadcast": 0.0, "frac_nothing": 0.0,
                "frac_teleport": 0.0, "frac_listen": 0.0, "frac_move": 0.0,
                "frac_reproduce": 0.0, "links": []}

    def snapshot(self) -> dict:
        """JSON-ready picture of world 0 -- alive ants only."""
        W = self.cfg.world_size
        alive0 = self.alive[0].detach().to("cpu")
        idx = torch.nonzero(alive0, as_tuple=False).flatten()
        pos0 = self.pos[0].detach().to("cpu")[idx]
        en0 = self.energy[0].detach().to("cpu")[idx]
        act0 = self.last_actions[0].detach().to("cpu")[idx]

        food0 = self.food[0].detach().to("cpu")
        nz = torch.nonzero(food0 > 0, as_tuple=False)
        food_list = [[int(x), int(y), float(food0[x, y])] for x, y in nz.tolist()]

        return {
            "world_size": W,
            "ants": {"pos": pos0.tolist(), "energy": en0.tolist(), "action": act0.tolist()},
            "food": food_list,
            "links": self.last_info.get("links", []),
            "metrics": {k: v for k, v in self.last_info.items() if k != "links"},
            "energy_max": self.cfg.energy_max,
            "max_food": self.cfg.max_food_size,
            "ecosystem": self.cfg.ecosystem,
        }
