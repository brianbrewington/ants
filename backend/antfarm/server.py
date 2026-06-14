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
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from .config import SimConfig
from .env import AntWorld
from .policies import make_policy

app = FastAPI(title="Communicating Ants")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# Params that can't change without re-allocating tensors -> require a reset.
STRUCTURAL = {"world_size", "n_ants", "n_worlds", "device", "seed"}


class Simulation:
    """Holds the one live world + the loop knobs. One per server process."""

    def __init__(self):
        self.cfg = SimConfig()
        self.env = AntWorld(self.cfg)
        self.policy = make_policy("heuristic")
        self.running = False
        self.steps_per_frame = 1   # env steps advanced between streamed frames
        self.fps = 30

    def reset(self, cfg: SimConfig | None = None):
        if cfg is not None:
            self.cfg = cfg
        self.env = AntWorld(self.cfg)

    def set_policy(self, name: str):
        self.policy = make_policy(name)

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

    def advance(self):
        for _ in range(max(1, self.steps_per_frame)):
            actions = self.policy(self.env)
            self.env.step(actions)


SIM = Simulation()


@app.get("/api/config")
def get_config():
    return JSONResponse({"config": SIM.cfg.to_dict(), "policy": SIM.policy.name})


async def _handle_control(ws: WebSocket, sim: Simulation):
    """Consume control messages from the client until it disconnects."""
    while True:
        msg = await ws.receive_json()
        t = msg.get("type")
        if t == "start":
            sim.running = True
        elif t == "pause":
            sim.running = False
        elif t == "reset":
            cfg = SimConfig.from_dict(msg.get("config", sim.cfg.to_dict()))
            sim.reset(cfg)
        elif t == "config":
            sim.update_config(msg.get("config", {}))
        elif t == "policy":
            sim.set_policy(msg.get("name", "heuristic"))
        elif t == "speed":
            sim.steps_per_frame = int(msg.get("steps_per_frame", sim.steps_per_frame))
            sim.fps = int(msg.get("fps", sim.fps))


async def _stream_frames(ws: WebSocket, sim: Simulation):
    """Advance the sim and push snapshots at the target frame rate."""
    while True:
        if sim.running:
            sim.advance()
        await ws.send_json({"type": "frame", "snapshot": sim.env.snapshot(),
                            "policy": sim.policy.name, "config": sim.cfg.to_dict()})
        await asyncio.sleep(1.0 / max(1, sim.fps))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    # Send an initial frame immediately so the canvas isn't blank.
    await ws.send_json({"type": "frame", "snapshot": SIM.env.snapshot(),
                        "policy": SIM.policy.name, "config": SIM.cfg.to_dict()})
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
