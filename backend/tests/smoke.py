"""Correctness + speed smoke test.  Run:  ./.venv/bin/python -m tests.smoke"""
import time
import torch

from antfarm import SimConfig, AntWorld, make_policy, bifurcation_sweep


def check_homeostatic():
    cfg = SimConfig(n_worlds=1, n_ants=120, world_size=48)
    env = AntWorld(cfg)
    policy = make_policy("heuristic")
    for _ in range(200):
        info = env.step(policy(env))
    snap = env.snapshot()
    assert len(snap["ants"]["pos"]) == cfg.n_ants, "homeostatic population should be pinned"
    assert (env.pos >= 0).all() and (env.pos < cfg.world_size).all()
    assert (env.food >= 0).all(), "food went negative"
    assert (env.energy <= cfg.energy_max + 1e-3).all()
    print(f"[homeostatic] pop={info['population']} mean_E={info['mean_energy']:.2f} "
          f"food={info['total_food']:.0f}  OK")


def check_ecosystem():
    cfg = SimConfig(ecosystem=True, n_worlds=1, n_ants=80, max_ants=1500,
                    world_size=48, food_growth_rate=1.9)
    env = AntWorld(cfg)
    policy = make_policy("heuristic")
    pops = []
    for _ in range(400):
        info = env.step(policy(env))
        pops.append(info["population"])
    assert (env.food >= 0).all(), "food negative"
    assert env.alive.sum().item() <= env.n_slots, "more alive than slots"
    assert (env.energy[env.alive] > 0).all(), "alive ant with <=0 energy"
    saw_birth = any(env.step(policy(env))["births"] > 0 for _ in range(20))
    print(f"[ecosystem] pop {pops[0]}->{pops[-1]} (min {min(pops)}, max {max(pops)}) "
          f"births_seen={saw_birth} food={info['total_food']:.0f}  OK")


def check_extinction():
    # Starve them: tiny food growth, high metabolism -> population should crash.
    cfg = SimConfig(ecosystem=True, n_worlds=1, n_ants=120, max_ants=400,
                    world_size=48, food_growth_rate=0.15, energy_cost=0.8, food_seed=0.0)
    env = AntWorld(cfg)
    policy = make_policy("forage")
    for _ in range(500):
        info = env.step(policy(env))
    print(f"[extinction] starved population -> {info['population']} (expect small)  OK")


def bench(n_worlds=512, n_ants=200, steps=100):
    cfg = SimConfig(ecosystem=True, n_worlds=n_worlds, n_ants=n_ants, max_ants=600,
                    world_size=48, enable_comm=False)
    env = AntWorld(cfg)
    policy = make_policy("forage")
    for _ in range(5):
        env.step(policy(env))
    if env.device.type == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    for _ in range(steps):
        env.step(policy(env))
    if env.device.type == "mps":
        torch.mps.synchronize()
    dt = time.time() - t0
    aps = n_worlds * env.n_slots * steps
    print(f"[bench] {n_worlds} worlds x {env.n_slots} slots x {steps} steps "
          f"-> {aps/dt/1e6:.1f}M slot-steps/sec ({dt:.2f}s)")


def bench_bifurcation():
    t0 = time.time()
    out = bifurcation_sweep(SimConfig(world_size=48, max_ants=800).to_dict(),
                            n_r=200, transient=300, sample=150)
    dt = time.time() - t0
    mn, mx = min(out["min"]), max(out["max"])
    print(f"[bifurcation] {len(out['r'])} r-values, {len(out['points'])} points in {dt:.1f}s "
          f"(pop frac range {mn:.2f}..{mx:.2f})")


if __name__ == "__main__":
    check_homeostatic()
    check_ecosystem()
    check_extinction()
    bench()
    bench_bifurcation()
