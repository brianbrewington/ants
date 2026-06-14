"""WebSocket control handling: hostile/malformed messages must be safe."""
from antfarm import SimConfig
from antfarm.server import Simulation, _apply_control


def _sim():
    sim = Simulation()
    sim.reset(SimConfig(device="cpu", world_size=16, n_ants=10))  # small + CPU for speed
    return sim


def test_speed_controls_clamped():
    sim = _sim()
    _apply_control(sim, {"type": "speed", "steps_per_frame": 10**9, "fps": 10**9})
    assert 1 <= sim.steps_per_frame <= 50
    assert 1 <= sim.fps <= 60


def test_reset_null_config_does_not_crash():
    sim = _sim()
    sim.running = True
    _apply_control(sim, {"type": "reset", "config": None})
    assert sim.running is False                # reset pauses (authoritative)
    assert sim.cfg.world_size >= 2


def test_partial_reset_preserves_unspecified_knobs():
    sim = _sim()
    _apply_control(sim, {"type": "reset", "config": {"world_size": 32, "device": "cpu"}})
    _apply_control(sim, {"type": "reset", "config": {"energy_cost": 0.9, "device": "cpu"}})
    assert sim.cfg.world_size == 32           # not reverted to the dataclass default
    assert abs(sim.cfg.energy_cost - 0.9) < 1e-9


def test_unknown_message_is_ignored():
    sim = _sim()
    before = sim.cfg.to_dict()
    _apply_control(sim, {"type": "nonsense"})
    assert sim.cfg.to_dict() == before


def test_advance_returns_frame_metrics_without_mutating_env():
    sim = _sim()
    sim.steps_per_frame = 5
    n0 = sim.env.step_count
    frame = sim.advance()
    assert sim.env.step_count == n0 + 5                 # advanced the right number of steps
    assert {"births", "deaths", "food_eaten", "population"} <= set(frame)
    # frame event counts are sums over the 5 steps, so >= the env's last single step
    assert frame["deaths"] >= sim.env.last_info["deaths"]
