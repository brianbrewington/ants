# Communicating Ants

A revival of a 1996–97 Dartmouth grad-school project, rebuilt for modern
hardware as a **teaching journey into multi-agent reinforcement learning**.

Digital ants roam a toroidal world looking for food. They can move, eat, and —
crucially — **communicate** where food is. The original research question still
stands: *when does it ever pay for a selfish agent to help others?* The original
project found cooperation hard to sustain (self-interested learning drives
communication out). This rebuild walks, lesson by lesson, from that failure to
the modern techniques that fix it.

The `original_code/` directory preserves the 1997 MATLAB sources, write-ups, and
data, untouched — the heritage this is built on.

## Architecture

```
backend/          Python · PyTorch (Apple MPS GPU) · FastAPI + WebSocket
  antfarm/
    config.py     all the simulation knobs (the old GUI sliders), typed + clamped
    env.py        the vectorized, GPU-resident world: shared mechanics + orchestration
    regimes.py    the 3 world models as strategies (homeostatic / logistic / nutrient)
    snapshot.py   world-0 -> JSON (presentation, kept out of the core)
    contracts.py  typed seams (StepInfo, Observation)
    policies.py   "brains": random + heuristic + forage; learned ones later
    bifurcation.py parallel-worlds sweep over food growth rate r
    server.py     runs the sim server-side, streams frames, owns frame metrics
    learning/     RL scaffolding (Learner / rewards / TabularQLearner stub)
  tests/          pytest: invariants, characterization goldens, regime/mask units
frontend/         React + TypeScript + Vite · canvas renderer · live charts
docs/             ARCHITECTURE.md + lessons/ (read alongside the code)
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the design decisions and seams.

The simulation runs **on the GPU, server-side**. The browser is a thin viewer:
it opens a WebSocket, receives ~30 world snapshots/sec, and sends back control
messages (start/pause/reset, knob changes, policy choice). This is the modern
analogue of the original MATLAB GUI — sliders and a START button — but with the
compute decoupled from the display.

## Quick start

**1. Backend** (Python 3.11+):

```bash
cd backend
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/uvicorn antfarm.server:app --port 8000
```

**2a. Frontend — dev mode** (hot reload, runs on :5173, talks to :8000):

```bash
cd frontend
npm install
npm run dev
# open http://localhost:5173
```

**2b. Frontend — production** (the backend serves the built app):

```bash
cd frontend && npm run build      # emits frontend/dist
# then just open http://localhost:8000  (uvicorn serves dist/)
```

**Sanity-check the world without a browser:**

```bash
cd backend && ./.venv/bin/python -m tests.smoke
```

## Development

```bash
cd backend
./.venv/bin/pip install -r requirements-dev.txt   # pytest + ruff
./.venv/bin/pytest          # invariants: food conservation, torus, death/respawn,
                            # reproduction slot-assignment, reproducibility
./.venv/bin/ruff check antfarm tests
```

Tests run on CPU so they match CI. The GitHub Actions workflow (ruff + pytest
for the backend, typecheck/build for the frontend) is in [`ci/ci.yml`](ci/ci.yml)
— see [`ci/README.md`](ci/README.md) for the one step to activate it (it's not
under `.github/workflows/` yet because pushing workflow files needs a
`workflow`-scoped token). Frontend formatting: `npx prettier` (config in
`frontend/.prettierrc.json`).

Runs are **reproducible**: a given `seed` fully determines a run (same seed +
same policy ⇒ identical trajectory). All client inputs are clamped in
`SimConfig.__post_init__` and the sweep endpoint, so no request can allocate an
absurd tensor.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the module map and the
non-obvious design decisions (slot pool, the MPS lock, food conservation), and
`backend/antfarm/learning/` for the RL stubs that the next lessons fill in.

## The lessons

| # | Title | What you learn |
|---|-------|----------------|
| 0 | [The World](docs/lessons/lesson-0-the-world.md) | The vectorized env + viz. No learning — the substrate. |
| 0.5 | [A Living Ecosystem](docs/lessons/lesson-0.5-ecology.md) | Renewable (logistic) food + free population → stability, boom-bust, extinction, and a live **bifurcation diagram** from parallel worlds. |
| 0.6 | [Closed Nutrient Loop](docs/lessons/lesson-0.6-nutrient-loop.md) | Food grows from a **conserved nutrient pool** (no hard cap, zeroed cells regrow, ants fertilize the soil). Mass-conserving; high-`r` cycles where logistic went extinct. |
| 1 | Tabular Q-learning | State, action, reward, the Bellman update — the original method, done right. |
| 2 | Deep Q (DQN) | Function approximation; reproduce the original "communication dies" failure. |
| 3 | The altruism fix | Team rewards + credit assignment (CTDE) so cooperation can pay. |
| 4 | Learned communication | Differentiable messages — let the ants invent what to say. |

You are here: **Lesson 1** — tabular Q-learning that visibly learns to survive
(headless training; ~37% fewer deaths than random). Lesson 0.x worlds remain in
the UI (World model toggle).

## Credits

Original concept & 1997 implementation: Brian Brewington (Dartmouth College).
Modern rebuild: a collaboration between Brian and Claude.
