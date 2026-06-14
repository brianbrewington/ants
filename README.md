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
    config.py     all the simulation knobs (the old GUI sliders), typed
    env.py        the vectorized, GPU-resident world  (no per-ant loops)
    policies.py   "brains": random + heuristic for now; learned ones later
    server.py     runs the sim server-side, streams frames to the browser
  tests/smoke.py  correctness + speed benchmark
frontend/         React + TypeScript + Vite · canvas renderer · live charts
docs/lessons/     lab notes — read these alongside the code
```

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

## The lessons

| # | Title | What you learn |
|---|-------|----------------|
| 0 | [The World](docs/lessons/lesson-0-the-world.md) | The vectorized env + viz. No learning — the substrate. |
| 0.5 | [A Living Ecosystem](docs/lessons/lesson-0.5-ecology.md) | Renewable (logistic) food + free population → stability, boom-bust, extinction, and a live **bifurcation diagram** from parallel worlds. |
| 1 | Tabular Q-learning | State, action, reward, the Bellman update — the original method, done right. |
| 2 | Deep Q (DQN) | Function approximation; reproduce the original "communication dies" failure. |
| 3 | The altruism fix | Team rewards + credit assignment (CTDE) so cooperation can pay. |
| 4 | Learned communication | Differentiable messages — let the ants invent what to say. |

You are here: **Lesson 0.5**. Toggle **World model → Ecosystem** in the UI.

## Credits

Original concept & 1997 implementation: Brian Brewington (Dartmouth College).
Modern rebuild: a collaboration between Brian and Claude.
