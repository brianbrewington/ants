"""
FastAPI + WebSocket server.

The simulation runs *here*, on the GPU, server-side. The browser is a thin
viewer: it opens a WebSocket, receives a stream of world snapshots (~30/sec),
and sends back control messages (start / pause / reset / knob changes / policy
choice). This is the modern analogue of the 1997 MATLAB GUI -- the sliders and
the START button, but decoupled from the compute.

Run (dev):
    ./.venv/bin/uvicorn antfarm.server:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .bifurcation import bifurcation_sweep
from .config import SimConfig
from .contracts import FrameMetrics
from .env import ACTION_NAMES, AntWorld
from .learning import TabularQLearner, survival_food_reward
from .policies import make_policy

app = FastAPI(title="Communicating Ants")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Params that can't change without re-allocating tensors -> require a reset.
STRUCTURAL = {"world_size", "n_ants", "n_worlds", "device", "seed", "ecosystem",
              "max_ants", "food_model"}

# A known-good food/energy economy for the bifurcation sweep, used when the live
# sim isn't currently in ecosystem mode (otherwise the sweep inherits homeostatic
# food, which is so abundant the population just pegs at the slot cap for every r).
# Mirrors frontend/src/presets.ts ECOSYSTEM_PRESET (minus structural fields).
ECO_SWEEP_ECONOMY = {
    "world_size": 48, "energy_max": 20.0, "energy_cost": 0.5, "bite_size": 3.0,
    "max_food_size": 12.0, "food_density": 0.06, "food_diffusion": 0.0,
    "food_seed": 0.06, "birth_threshold": 0.6, "birth_cost": 0.5,
}


class Simulation:
    """Holds the one live world + the loop knobs. One per server process.

    The "brain" is either a stateless hand policy (heuristic/random/forage) or a
    stateful learner (tabular_q). The learner has a lifecycle: it's bound to the
    env, learns online as the sim runs (when `learning` is on), and its Q-table is
    preserved across non-structural resets but rebuilt when the world's shape
    changes. (This is the trainer/registry seam the architecture review flagged.)
    """

    def __init__(self):
        self.cfg = SimConfig()
        self.env = AntWorld(self.cfg)
        self.running = False
        self.steps_per_frame = 1   # env steps advanced between streamed frames
        self.fps = 30
        self.brain = "heuristic"
        self.policy = make_policy("heuristic")
        self.learner: TabularQLearner | None = None
        self.learning = True       # learner: learn (True) vs freeze/eval (False)
        self.last_reward = 0.0     # telemetry: mean reward over the last frame

    def reset(self, cfg: SimConfig | None = None):
        if cfg is not None:
            self.cfg = cfg
        self.env = AntWorld(self.cfg)
        if self.learner is not None:                       # rebind learner to new env
            self._make_learner(preserve=True)

    def set_brain(self, name: str):
        self.brain = name
        if name == "tabular_q":
            self._make_learner(preserve=self.learner is not None)
        else:
            self.learner = None
            self.policy = make_policy(name)

    def _make_learner(self, preserve: bool):
        prev_q = self.learner.Q if (preserve and self.learner is not None) else None
        self.learner = TabularQLearner(self.env)
        if prev_q is not None and prev_q.shape == self.learner.Q.shape:
            self.learner.Q = prev_q.to(self.learner.device)   # keep what it learned

    def reset_learner(self):
        if self.learner is not None:
            self._make_learner(preserve=False)             # wipe the Q-table, start fresh

    def learner_telemetry(self) -> dict | None:
        """Small enough to stream every frame: the whole Q-table (states x actions),
        labels, exploration rate, and recent reward -- the learner's whole mind."""
        if self.learner is None:
            return None
        actions = [("migrate" if a == "teleport" else a) for a in ACTION_NAMES]
        return {
            "q_table": self.learner.Q.detach().to("cpu").tolist(),
            "states": self.learner.state_labels(),
            "actions": actions,
            "epsilon": float(self.learner.epsilon),
            "reward": self.last_reward,
            "learning": self.learning,
        }

    def update_config(self, updates: dict):
        """Apply knob changes. Structural ones rebuild the world; the rest are
        live-edited in place (the env reads cfg fields every step)."""
        new_cfg = self.cfg.with_updates(
            **{k: v for k, v in updates.items() if k in SimConfig.__dataclass_fields__}
        )
        structural_changed = any(
            getattr(new_cfg, k) != getattr(self.cfg, k) for k in STRUCTURAL
        )
        self.cfg = new_cfg
        if structural_changed:
            self.reset(new_cfg)
        else:
            self.env.cfg = new_cfg  # live knobs take effect next step

    def advance(self) -> FrameMetrics:
        """Advance steps_per_frame steps and RETURN frame metrics. Event counts
        (births/deaths/food_eaten) are summed across the frame so charts aren't
        undercounted at speed>1; state quantities come from the last step. The env
        owns world state; the server owns frame aggregation -- no mutating env."""
        n = max(1, self.steps_per_frame)
        last = self.env.last_info
        births = deaths = 0
        eaten = 0.0
        reward_sum = 0.0
        frac_sum: dict[str, float] = {}
        for _ in range(n):
            if self.learner is not None:
                last = self._learner_step()
                reward_sum += self.last_reward
            else:
                last = self.env.step(self.policy(self.env))
            births += last["births"]
            deaths += last["deaths"]
            eaten += last["food_eaten"]
            for k, v in last.items():        # action mix averaged over the frame
                if k.startswith("frac_"):
                    frac_sum[k] = frac_sum.get(k, 0.0) + v
        if self.learner is not None:
            self.last_reward = reward_sum / n
        frame = {k: v for k, v in last.items() if k != "links"}
        frame.update(births=births, deaths=deaths, food_eaten=eaten)
        frame.update({k: s / n for k, s in frac_sum.items()})
        return frame

    def _learner_step(self) -> dict:
        """One live learning step: act (explore unless frozen), full env.step (we
        need metrics for the viz), then learn from the transition. Mirrors
        learning/trainer.py but keeps metrics + reward telemetry."""
        env, learner = self.env, self.learner
        s1 = learner.encode(env)
        actions = learner.act(env, greedy=not self.learning)
        prev_energy = env.energy.clone()
        info = env.step(actions)                       # full step (metrics for the UI)
        if self.learning:
            reward = survival_food_reward(env, prev_energy, env.last_died)
            learner.update(s1, actions, reward, learner.encode(env), env)
            self.last_reward = float(reward.mean().item())
        return info


SIM = Simulation()

# PyTorch's MPS (Metal) backend is NOT thread-safe: two threads issuing GPU ops
# at once segfaults the Metal command stream. We therefore funnel ALL env/GPU
# work through this single lock, so the live per-frame step and the bifurcation
# sweep can never overlap. (Also why the sweep endpoint is async, not sync --
# a sync handler would run in a worker thread.)
SIM_LOCK = asyncio.Lock()


@app.get("/api/config")
def get_config():
    return JSONResponse({"config": SIM.cfg.to_dict(), "policy": SIM.policy.name})


@app.get("/api/bifurcation")
async def bifurcation(r_min: float = 0.3, r_max: float = 3.0, n_r: int = 200,
                      transient: int = 400, sample: int = 240, runs_per_r: int = 1):
    """Run the parallel-worlds sweep over food growth rate r.

    Declared async and run directly on the event-loop thread *on purpose*: a sync
    endpoint would be dispatched to a FastAPI worker thread, and PyTorch's MPS
    backend must be driven from the thread that initialized it. The sweep is only
    a few seconds, so briefly blocking the loop (pausing the live stream) is fine
    and intentional -- a background job would reintroduce the cross-thread MPS
    segfault that single-threading avoids.
    It inherits the current sim's food/energy knobs; the sweep raises the slot cap
    internally so FOOD, not the cap, limits the population."""
    # Clamp every knob: the sweep allocates [n_r, max_ants] tensors and runs
    # (transient+sample) steps, so unbounded inputs are an OOM / GPU-hog footgun.
    n_r = max(10, min(512, n_r))
    runs_per_r = max(1, min(12, runs_per_r))
    transient = max(0, min(5000, transient))
    sample = max(10, min(3000, sample))
    r_min = max(0.0, min(4.0, r_min))
    r_max = max(r_min + 1e-3, min(4.0, r_max))
    # Bound the total world count (n_r * runs_per_r): each is a full simulation.
    if n_r * runs_per_r > 4000:
        runs_per_r = max(1, 4000 // n_r)

    base = SIM.cfg.to_dict()
    if not base.get("ecosystem"):
        base.update(ECO_SWEEP_ECONOMY)  # ensure a food-limited regime, not flat
    # Cap the sweep's world size independently of the live sim: the sweep runs
    # many worlds, so a large live world_size would blow up worlds*W^2 food tensors.
    base["world_size"] = min(int(base.get("world_size", 48)), 64)
    async with SIM_LOCK:                  # never run GPU work alongside the live step
        out = bifurcation_sweep(base, r_min=r_min, r_max=r_max, n_r=n_r,
                                transient=transient, sample=sample, runs_per_r=runs_per_r)
    return JSONResponse(out)


def _as_dict(v) -> dict:
    """Coerce a client-supplied config payload to a dict (None / wrong type -> {})."""
    return v if isinstance(v, dict) else {}


def _apply_control(sim: Simulation, msg: dict) -> None:
    t = msg.get("type")
    if t == "start":
        sim.running = True
    elif t == "pause":
        sim.running = False
    elif t == "reset":
        # Merge over the current config so a partial (or null) reset preserves
        # knobs the client didn't specify, and `config: null` can't crash us.
        merged = {**sim.cfg.to_dict(), **_as_dict(msg.get("config"))}
        sim.reset(SimConfig.from_dict(merged))
        sim.running = False                      # reset always pauses (authoritative)
    elif t == "config":
        sim.update_config(_as_dict(msg.get("config")))
    elif t == "policy":
        sim.set_brain(msg.get("name", "heuristic"))
    elif t == "learning":                        # learner: learn vs freeze/eval
        sim.learning = bool(msg.get("on", True))
    elif t == "reset_learner":
        sim.reset_learner()
    elif t == "speed":
        if "steps_per_frame" in msg:
            sim.steps_per_frame = max(1, min(50, int(msg["steps_per_frame"])))
        if "fps" in msg:
            sim.fps = max(1, min(60, int(msg["fps"])))


async def _handle_control(ws: WebSocket, sim: Simulation):
    """Consume control messages until the client disconnects. A single malformed
    message must never tear down the stream, so each is parsed defensively."""
    while True:
        msg = await ws.receive_json()
        if not isinstance(msg, dict):
            continue
        try:
            _apply_control(sim, msg)
        except (KeyError, TypeError, ValueError):
            continue  # ignore bad control message; keep streaming


async def _stream_frames(ws: WebSocket, sim: Simulation):
    """Advance the sim and push snapshots at the target frame rate."""
    while True:
        async with SIM_LOCK:                 # serialize all GPU access (see SIM_LOCK)
            frame_metrics = sim.advance() if sim.running else None
            snapshot = sim.env.snapshot()
        if frame_metrics is not None:
            snapshot["metrics"] = frame_metrics   # frame-aggregated, server-owned
        await ws.send_json({"type": "frame", "snapshot": snapshot, "running": sim.running,
                            "policy": sim.brain, "learner": sim.learner_telemetry(),
                            "config": sim.cfg.to_dict()})
        await asyncio.sleep(1.0 / max(1, sim.fps))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    # Send an initial frame immediately so the canvas isn't blank.
    await ws.send_json({"type": "frame", "snapshot": SIM.env.snapshot(), "running": SIM.running,
                        "policy": SIM.brain, "learner": SIM.learner_telemetry(),
                        "config": SIM.cfg.to_dict()})
    producer = asyncio.create_task(_stream_frames(ws, SIM))
    consumer = asyncio.create_task(_handle_control(ws, SIM))
    try:
        await asyncio.gather(producer, consumer)
    except WebSocketDisconnect:
        pass
    finally:
        producer.cancel()
        consumer.cancel()


# In production we serve the built React app. In dev the frontend runs on the
# Vite dev server, so this directory may not exist yet.
_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
if _DIST.exists():
    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="static")
