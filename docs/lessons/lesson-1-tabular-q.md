# Lesson 1 — Tabular Q-learning

> Goal: replace the hand-written brain with one that *learns*, using the original
> project's actual method — a Q-table — done correctly, and fixing the one thing
> the 1996 reward was missing: **a reason not to die.**

## The idea, in plain words

Keep a table: for every *state* an ant can be in, and every *action* it could
take, store a number — "how good is doing this here?" That's `Q[state, action]`.
Acting is then easy: look up the row for your state, do the action with the
highest number (most of the time).

Learning is the interesting part. After you act and see what happened, you nudge
that number toward what experience says it *should* have been — the **Bellman
target**:

```
target  = reward_you_just_got  +  γ · (best Q value available in the next state)
Q[s,a] += lr · (target − Q[s,a])
```

In English: *the value of acting now is the reward you just got, plus the
discounted value of the best thing you can do next.* Repeat over millions of
transitions and the table fills in with genuine foresight. This is Q-learning
(Watkins, 1989) — exactly what your `qupdate.m` did. Ours just runs it over every
ant in every parallel world at once (`learning/tabular.py:update`), and only ever
picks **legal** actions (the `action_mask` from Lesson 0.5's refactor).

## The fix: dying has to cost something

The 1996 reward only rewarded eating. So a learned policy had every reason to
nibble and none to *survive* — starving was free. We give the env a per-step
death signal (`env.last_died`) and a reward that says: *reward = energy you
gained, but a big penalty if you died this step* (`learning/rewards.py`). That one
change is the lesson's headline.

## How it runs

A `Trainer` (`learning/trainer.py`) owns the loop; the `Learner` owns act/update:

```
s1 = encode(env)              # (energy bin, on food?, heard food?)
a  = act(env)                 # ε-greedy over legal actions
prev_energy = env.energy
env.step(a)
r  = reward(env, prev_energy, env.last_died)
s2 = encode(env)
learner.update(s1, a, r, s2, env)
```

Run headless across dozens of worlds → thousands of transitions per step. This is
also the seam the architecture review asked for: a learner has a *lifecycle* a
trainer drives, not "just a policy you swap in." (Replay buffers and centralized
critics come in later lessons; the loop shape stays.)

## Does it work?

Yes — trained over 64 worlds, then evaluated greedily against a random baseline:

| policy | deaths / 1k ant-steps | mean energy |
|---|---|---|
| random | ~16.7 | 10.2 |
| **learned** | **~10.5** | **12.0** |

**~37% fewer deaths, +18% energy** — the ant learned to eat when it's on food and
hungry, and to stop squandering itself. (`test_learning.py` pins both the update
arithmetic and that learning beats random.)

## What's still weak — and what it sets up

The state is coarse: `(energy bin) × (on food?) × (heard food?)` — 40 states. The
ant **can't see where food is**, only whether it's standing on some, so it can't
really learn to *seek* food; its wins are mostly "eat now" and "don't waste
moves." That ceiling is the motivation for **Lesson 2 (Deep Q)**: swap the table
for a small neural net that consumes a richer observation, so the ant can learn to
*navigate* — and there we'll reproduce the original project's real failure:
with a per-ant reward, **communication still doesn't pay**, which is what forces
the team-reward work of Lesson 3.
