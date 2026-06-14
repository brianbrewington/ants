# Lesson 1 — Tabular Q-learning  (UPCOMING)

> Status: **planned**. The scaffold exists in `backend/antfarm/learning/` —
> `TabularQLearner.encode()` and `.act()` are written; `.update()` is the TODO we
> fill in this lesson. This file is the plan, not yet the finished write-up.

## Goal

Replace the hand-written brain with one that *learns*, using the original
project's actual method — a Q-table — but done correctly. By the end, the learned
policy should visibly beat the heuristic baseline on the live charts, and we'll
have introduced the core RL vocabulary: **state, action, reward, value, the
Bellman update, exploration**.

## The one thing the 1997 version got wrong

The original reward only rewarded eating; **dying cost nothing**. So ants learned
to nibble when food was under them but had no pressure to *survive*. We fix that
with `survival_food_reward` (already stubbed): reward = energy gained, with a big
negative on death. That single change is the lesson's headline.

## Plan

1. **State** (`encode`, done): discretize each ant into
   `(energy bin) × (on food?) × (heard food?)` — compact and legible, the spirit
   of the original `getstate.m`.
2. **Action**: the same 7-action set; epsilon-greedy selection (`act`, done).
3. **Reward**: `survival_food_reward` (done).
4. **Update** (the TODO): vectorized Q-learning over the whole `[n_worlds,
   n_slots]` batch at once:
   ```
   target = reward + γ · max_a' Q[s2, a']
   Q[s1, a] += lr · (target − Q[s1, a])
   ```
   Use scatter/index_put_; mask out dead slots; decay epsilon over time.
5. **Wire it in**: register the learner as a selectable "brain" in the server so
   you can watch it learn live, and add a "learning on/off" + epsilon readout.

## What to watch

- Mean energy and survival climbing above the heuristic baseline.
- The action mix shifting toward eating-when-on-food and away from wasted moves.
- (Foreshadowing Lesson 2) communication staying near zero — because with a
  per-ant reward, talking still doesn't pay. That failure motivates Lessons 3–4.

## Why the batch layout matters here

`encode/act/update` all operate on `[n_worlds, n_slots]`, so the learner trains
on every ant in every parallel world simultaneously — thousands of transitions
per step. That's "parallel rollouts," already built into the env.
