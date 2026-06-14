"""Lesson 1: tabular Q-learning — the update is correct, and it actually learns."""
import torch

from antfarm import AntWorld, SimConfig
from antfarm.learning import TabularQLearner, Trainer
from antfarm.policies import RandomPolicy


def test_bellman_update_moves_Q_toward_target():
    # One ant, hand-built transition, so the arithmetic is checkable:
    #   target = r + gamma * max_a' Q[s2]; Q[s1,a] += lr * (target - Q[s1,a])
    env = AntWorld(SimConfig(device="cpu", n_ants=1, world_size=8))
    lr, gamma = 0.2, 0.9
    learner = TabularQLearner(env, lr=lr, gamma=gamma)
    learner.Q.zero_()
    learner.Q[5, 0] = 1.0                         # so max_a' Q[s2=5] = 1.0
    s1, a = torch.tensor([[3]]), torch.tensor([[1]])
    r, s2 = torch.tensor([[2.0]]), torch.tensor([[5]])
    learner.update(s1, a, r, s2, env)
    target = 2.0 + gamma * 1.0                    # 2.9
    assert abs(float(learner.Q[3, 1]) - lr * target) < 1e-5   # 0.2 * 2.9 = 0.58


def test_action_choices_are_always_legal():
    env = AntWorld(SimConfig(device="cpu", ecosystem=True, n_ants=10, max_ants=40, world_size=10))
    learner = TabularQLearner(env, epsilon=1.0)   # force exploration
    mask = env.action_mask()
    a = learner.act(env)
    chosen_legal = mask.gather(-1, a.unsqueeze(-1)).squeeze(-1)
    assert bool(chosen_legal.all()), "learner chose an illegal action"


def _death_rate(env, act_fn, steps=200):
    n = env.cfg.n_worlds * env.n_slots
    deaths = 0
    for _ in range(steps):
        env.step(act_fn(env), light=True)
        deaths += int(env.last_died.sum())
    return deaths / (n * steps)


def test_learning_reduces_deaths_vs_random():
    cfg = SimConfig(device="cpu", seed=1, n_worlds=16, n_ants=60, world_size=16, energy_cost=0.4)
    rand_dr = _death_rate(AntWorld(cfg), RandomPolicy())

    env = AntWorld(cfg)
    learner = TabularQLearner(env, lr=0.2, gamma=0.9, epsilon=0.25)
    Trainer(env, learner).train(2500)

    learner.epsilon = 0.0
    learned_dr = _death_rate(AntWorld(cfg), learner)
    assert learned_dr < rand_dr, f"learned {learned_dr:.4f} not better than random {rand_dr:.4f}"
