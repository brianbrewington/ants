# Lesson 0.6 — The Closed Nutrient Loop

> Goal: fix a flaw we *saw* in Lesson 0.5. In the logistic food model, a food cell
> that's grazed to exactly zero is an **absorbing state** — `r·0·(1−0/K) = 0`, so
> it can never grow back on its own. At high growth rate `r` the discrete logistic
> overshoots and crashes cells to zero faster than they're reseeded, and the whole
> food supply collapses → extinction. Counterintuitively, *more* growth killed the
> world. This lesson removes that failure at its root.

## The idea: grow food from a conserved resource

Instead of food appearing from a hardcoded carrying capacity `K`, food grows out
of a **nutrient substrate** `N` that sits in every cell — the "soil." Three
pools now hold all the mass in the world:

```
nutrient N  ──food grows from N──>  food F  ──ants eat──>  ant energy
    ^                                                          │
    └──────── ants return mass to the soil as they ───────────┘
              metabolize (and die)
```

Per step (`env._grow_food_nutrient`):
1. **Growth** on cells that already have food, limited by how much nutrient is
   locally left (a Monod term `N/(N+h)`), with the mass moved `N → F`. As food
   accumulates it depletes the local nutrient, which throttles further growth —
   a *soft, emergent* cap instead of a hard `K`.
2. **Germination**: a few random cells convert local nutrient into a fresh food
   patch. This is the mass-conserving replacement for Lesson 0.5's "spore rain,"
   and it's what makes a zeroed cell **non-absorbing** — food can always
   recolonize while nutrient remains beneath it.
3. **Nutrient diffusion** (optional) and **external inflow** (optional).

And the ants close the loop: every step a living ant **returns the energy it
burns to the nutrient pool** at its cell (`metabolism → N`), and whatever mass is
left at death goes back too. Ants fertilize the soil they walk on.

## Two payoffs

**1. Mass is conserved — a real, testable invariant.** With `nutrient_inflow = 0`
the world is *closed*: every term just moves mass between `N`, `F`, and `energy`,
so `ΣN + ΣF + Σenergy` is constant. `test_nutrient.py` asserts it holds to 0.1%
over hundreds of steps. (Raise inflow to open the system — "sunlight" — and total
mass grows; also tested.) The carrying capacity is no longer a magic number; it's
just *how much mass you put in the world.*

**2. High `r` cycles instead of collapsing.** Same experiment that killed the
logistic world, side by side:

| food growth `r` | logistic (0.5) | nutrient (0.6) |
|---|---|---|
| 2.5 | ~244 ants | ~87 ants (steadier) |
| **3.8** | **0 — extinct** | **~85 ants, cycling (26–193)** |

At `r = 3.8` the logistic model's food crashes to an absorbing zero and everyone
starves; the nutrient model just germinates food back out of the soil and the
population boom-busts around a healthy mean. The nutrient pool is the floor under
the collapse.

## Things to try

- **Watch the soil**: the new "Nutrient in soil" sparkline and "Food on grid"
  trace move in **opposition** — food blooms drain the soil, then metabolism and
  death refill it. That anti-phase is the loop breathing.
- **Push `r` to 4** and compare to the same `r` in Ecosystem mode (which goes
  extinct). The nutrient world should keep cycling.
- **Closed vs open**: at `inflow = 0` the population is capped by the fixed mass
  budget (`nutrient/cell`); add a little inflow and watch the ceiling rise.
- **Run the bifurcation sweep in nutrient mode** — the diagram should look
  different from the logistic one, especially at the high-`r` end (no absorbing
  extinction band).

## What this sets up

This is the world the RL lessons deserve: scarcity is real, the resource can't be
gamed by a modeling artifact, and (because dead ants enrich the soil) there's a
genuine spatial, temporal coupling between a group's fate and its environment.
That's fertile ground for the altruism question in Lessons 3–4. A natural next
refinement: give ants a structural **body mass** so carcasses become real fertile
hotspots (right now a starved ant has burned itself to ~0 before dying).
