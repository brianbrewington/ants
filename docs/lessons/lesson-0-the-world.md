# Lesson 0 — The World

> Goal: get the 1996 idea breathing again on modern hardware. No learning yet —
> just a living world we can watch, so that every later lesson has something
> real to improve.

## What you're looking at

A population of ants wanders a square, **toroidal** grid (walk off the right
edge, reappear on the left). Amber squares are food. Ants are dots, coloured by
the action they just took, and the green threads are live **communications**
between an ant that's broadcasting and one that's listening.

There is no brain here. Two hand-written policies drive the ants:

- **heuristic** — a sensible rule of thumb: *eat when you're on food; sometimes
  announce a food find; sometimes go listen for someone else's tip; otherwise
  wander.* It's lively and shows the communication machinery working.
- **random** — pure noise. Ants almost never eat on purpose, so they starve and
  respawn constantly. Watch the **deaths/frame** chart spike. This is the whole
  motivation for the rest of the project: *we need a brain.*

Neither policy *learns*. They're the baselines a learned policy will have to
beat in Lessons 1–4.

## The one idea that makes this fast: vectorization

The original MATLAB looped over ants, one at a time, every step. We never do.
Every ant attribute is a tensor with a batch layout:

```
pos     [n_worlds, n_ants, 2]     # continuous (x, y)
energy  [n_worlds, n_ants]
food    [n_worlds, world, world]
```

"Step every ant" becomes a handful of array operations that run in parallel on
the GPU (Apple's Metal/**MPS** backend on this Mac — there's no CUDA here). The
`n_worlds` dimension is a *batch of independent simulations*. For Lesson 0 we
run **one** world and watch it. But the same code already runs **hundreds** of
worlds at once — try the benchmark:

```
cd backend && ./.venv/bin/python -m tests.smoke
# 512 worlds x 120 ants x 100 steps  ->  millions of ant-steps/sec
```

That batch dimension is exactly what reinforcement learning will feed on:
"parallel rollouts" just means *let many worlds play at once and learn from all
of them.* You already have the hard part.

## The six actions

These are the original project's six, unchanged:

| id | action     | effect                                                        |
|----|------------|---------------------------------------------------------------|
| 0  | eat        | consume food in the current cell (if any, and not full)       |
| 1  | broadcast  | announce your location to nearby listeners                    |
| 2  | nothing    | rest                                                          |
| 3  | teleport   | jump to the destination you last heard about                  |
| 4  | listen     | adopt the nearest broadcaster's location as your destination  |
| 5  | randmove   | wander a short random step                                    |

Note how communication is a **two-step loop**: one ant *listens* (action 4) to
learn where food is, then *teleports* (action 3) there. That coupling — and
whether selfish agents ever find it worthwhile — is the research question we're
ultimately chasing.

## One subtlety worth seeing: eating without cheating

Many ants can share a cell. If each just took a full bite, they'd eat more food
than physically exists. So `env.step` computes the **total demand** at each cell,
compares it to what's there, and scales every ant's bite by the same fraction.
Food is conserved exactly (the smoke test asserts `food >= 0`). This kind of
"resolve contention with a vectorized reduction" pattern shows up everywhere in
batched simulation — it's worth reading `env.py`'s EAT section once.

## Knobs (the 1997 sliders, reborn)

The control panel edits the same `SimConfig` the env reads. Most knobs apply
**live** (next step); changing **ant count** or **world size** rebuilds the
world. Things to try:

- Crank **food density** down and watch competition (and deaths) rise.
- Widen **comm radius** and you'll see more green link threads.
- Switch to **random** and watch the world collapse into churn.

## What's next

- **Lesson 1 — Tabular Q-learning.** Replace the hand-written policy with the
  original project's actual method (a Q-table), done correctly, plus a death
  penalty so ants learn to *survive*, not just nibble.
- **Lesson 2 — Deep Q.** Swap the table for a small neural net, and reproduce
  the original failure: communication gets driven out of the policy.
- **Lesson 3 — The altruism fix.** Team rewards + proper credit assignment so
  cooperation can actually become rational.
- **Lesson 4 — Learned communication.** Stop hard-coding broadcast/listen; let
  the ants invent what to say.
