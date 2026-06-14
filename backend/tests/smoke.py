"""Quick correctness + speed smoke test for the vectorized world.

Run:  ./.venv/bin/python -m tests.smoke
"""
import time
import torch

from antfarm import SimConfig, AntWorld, make_policy


def check_single_world():
    cfg = SimConfig(n_worlds=1, n_ants=120, world_size=48)
    env = AntWorld(cfg)
    policy = make_policy("heuristic")
    print(f"device={env.device}")

    food_start = env.food.sum().item()
    for _ in range(200):
        actions = policy(env)
        info = env.step(actions)
    snap = env.snapshot()
    assert len(snap["ants"]["pos"]) == cfg.n_ants
    assert (env.pos >= 0).all() and (env.pos < cfg.world_size).all(), "positions left the torus"
    assert (env.food >= 0).all(), "food went negative -- eating contention bug"
    assert (env.energy <= cfg.energy_max + 1e-3).all(), "energy exceeded the cap"
    print(f"step={info['step']} mean_energy={info['mean_energy']:.2f} "
          f"deaths(last)={info['deaths']} food_on_grid={info['total_food']:.0f} "
          f"(was {food_start:.0f}) links={len(snap['links'])}")
    print("single-world checks passed.")


def bench_many_worlds(n_worlds=512, n_ants=120, steps=100):
    cfg = SimConfig(n_worlds=n_worlds, n_ants=n_ants, world_size=48)
    env = AntWorld(cfg)
    policy = make_policy("heuristic")
    # warmup (MPS compiles kernels lazily)
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
    ant_steps = n_worlds * n_ants * steps
    print(f"{n_worlds} worlds x {n_ants} ants x {steps} steps = {ant_steps:,} ant-steps "
          f"in {dt:.2f}s  ->  {ant_steps/dt/1e6:.1f}M ant-steps/sec")


if __name__ == "__main__":
    check_single_world()
    bench_many_worlds()
