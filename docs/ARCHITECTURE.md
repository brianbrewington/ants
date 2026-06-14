# Architecture

A map of the codebase for reviewers and future contributors. For the *why* and
the science, read the lesson notes in `docs/lessons/`.

## Big picture

```
browser (React SPA)  ──WebSocket──>  FastAPI server  ──>  AntWorld (PyTorch, GPU)
     ^  controls / knobs                   |  one sim per process
     └──────  ~30 frames/sec  ────────────┘
```

The simulation runs server-side on the GPU (Apple **MPS**; falls back to CPU).
The browser is a thin viewer: it streams world snapshots and sends control
messages. This mirrors the 1997 MATLAB GUI but decouples display from compute.

## Backend (`backend/antfarm/`)

| File | Responsibility |
|------|----------------|
| `config.py` | `SimConfig` dataclass — every knob, plus `__post_init__` validation/clamping. |
| `env.py` | `AntWorld` — the vectorized world: shared mechanics (eat/move/comm/reproduce) + orchestration. |
| `regimes.py` | The three world models as strategies (Homeostatic / Logistic / Nutrient). |
| `snapshot.py` | World-0 → JSON serialization (presentation, kept out of the core). |
| `contracts.py` | Typed seams: `StepInfo`, `Observation`. |
| `policies.py` | Hand-written brains: `Random`, `Heuristic`, `Forage`. Contract: `policy(env) -> actions`. |
| `bifurcation.py` | Parallel-worlds sweep over food growth rate `r` → bifurcation data. |
| `server.py` | FastAPI app: `/ws` stream, `/api/config`, `/api/bifurcation`; the GPU lock; frame-metric aggregation. |
| `learning/` | RL scaffolding for the upcoming lessons (stubs — see below). |
| `tests/` (`backend/tests/`) | `pytest` invariants, characterization goldens, regime/mask unit tests + `smoke.py`. |

### Core data model — the batch layout
Every ant attribute is a tensor shaped `[n_worlds, n_slots]` (or `…, 2` for
positions). We never loop over ants. `n_worlds` is a batch of independent worlds
(used by the bifurcation sweep and, later, RL rollouts). One step = a handful of
vectorized GPU ops.

### Key design decisions (the non-obvious bits a reviewer should know)
- **Variable population via a slot pool.** Ecosystem mode allocates `max_ants`
  columns and carries a boolean `alive` mask. Births fill dead slots; deaths
  clear the mask; dead slots are forced to `NOTHING` and filtered from the view.
  "No free slot" is a natural carrying capacity. Reproduction assigns offspring
  to free slots with a vectorized cumsum-rank trick — **no Python loop over
  worlds** (`env._reproduce`). Tested for collision-freeness in `test_env.py`.
- **Reproducibility.** One per-env `torch.Generator` (`env.gen`, on-device,
  seeded from `cfg.seed`) drives *all* randomness — initial state, movement,
  spores, respawn, births — and the hand policies draw from it too. Same seed +
  same policy ⇒ identical run (asserted by `test_seed_is_actually_reproducible`).
- **MPS is not thread-safe.** All GPU work is serialized through `SIM_LOCK`
  (`asyncio.Lock`), and `/api/bifurcation` is `async` (a sync handler would run
  in a worker thread and segfault Metal). See `docs/lessons/lesson-0.5` history.
- **Food conservation under contention.** Many ants can share a cell; `env.step`
  computes total demand per cell and scales each bite by `min(1, avail/demand)`
  so food is conserved exactly (`test_eat_does_not_create_energy`).
- **`light=True` step.** Skips per-step GPU→CPU metric syncs; used by the sweep
  so it runs in seconds instead of a minute.
- **Three world regimes, as strategies** (`regimes.py`). `step()`/`reset()` are
  pure orchestration; each regime owns its reset / metabolism / births-deaths /
  food-growth sequence:
  - **Homeostatic** (Lesson 0): food refills to a target, dead ants respawn —
    population pinned.
  - **Logistic** (Lesson 0.5): logistic food (rate `r`), diffusion off, spore
    rain; free population (boom-bust, extinction, collapse).
  - **Nutrient** (Lesson 0.6): food grows from a conserved nutrient pool;
    metabolism/death return mass to the soil. Closed (`inflow=loss=0`) conserves
    `N+F+energy`; open (`inflow>0` + `loss>0`) is dissipative and re-bifurcates.
- **Metrics ownership.** `env.step()` returns per-step `StepInfo`; the **server**
  sums event counts across a frame (`Simulation.advance`) and owns the frame
  metrics — the env is never mutated for streaming. Legal-action masking is a
  separate surface (`env.action_mask()`), an input for future learners.

### Input safety
`SimConfig.__post_init__` clamps every field (sizes capped to prevent OOM;
diffusion capped at the PDE stability limit). The sweep endpoint clamps
`n_r/transient/sample/r`. So no REST query or WS message can allocate an absurd
tensor or destabilize the food update.

## Frontend (`frontend/src/`)

| File | Responsibility |
|------|----------------|
| `App.tsx` | Layout, mode toggle (applies a preset reset), wiring. |
| `useSim.ts` | WebSocket hook: latest frame, metric history, reconnect, `send()`. |
| `AntCanvas.tsx` | Canvas renderer: food, ants (colour=action, size=energy), comm links. |
| `Controls.tsx` | The reborn slider panel; live vs structural knobs. |
| `Metrics.tsx` | Sparklines + action-mix bars (with tooltips). |
| `Bifurcation.tsx` | Runs `/api/bifurcation`, draws the diagram; tunable settle/sample/resolution. |
| `presets.ts` | Homeostatic vs Ecosystem parameter presets. |
| `types.ts` | Shared types + action ids/colours (mirror `env.py`). |

## How RL plugs in (the seam for Lessons 1–4)

`learning/` is scaffolding, not yet wired into the server:
- `Learner` (base.py) is a policy that also learns — same `callable(env)->actions`
  contract as the hand policies, plus `encode()`/`update()` hooks.
- `rewards.py` — `survival_food_reward` (adds the death penalty the 1997 code
  lacked); `team_reward` is a Lesson-3 stub (CTDE / VDN).
- `tabular.py` — `TabularQLearner`: `encode()`/`act()` implemented, `update()` is
  the Lesson-1 TODO.

A finished learner is just another policy, so it drops into `Simulation.policy`
unchanged; the training loop is documented in `learning/base.py`.
