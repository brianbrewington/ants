"""
Reinforcement-learning scaffolding for the upcoming lessons.

NOTHING HERE IS WIRED INTO THE LIVE SIM YET — these are deliberate stubs that
fix the *shape* of the RL code so each lesson has a clear seam to fill:

  base.py     — the Learner interface (a policy that also learns). It satisfies
                the same callable(env)->actions contract as the hand-written
                policies, so a finished learner drops straight into the server.
  rewards.py  — reward functions, including the death penalty the 1997 project
                never had, plus a team-reward stub for the altruism lesson.
  tabular.py  — Lesson 1: a modern, correct redo of the original qupdate.m
                Q-table. encode()/act() are implemented; update() is the TODO.

See docs/lessons/lesson-1-tabular-q.md for the plan.
"""

from .base import Learner
from .rewards import survival_food_reward
from .tabular import TabularQLearner

__all__ = ["Learner", "survival_food_reward", "TabularQLearner"]
