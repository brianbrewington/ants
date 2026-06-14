# Lesson 0.5 — A Living Ecosystem

> Goal: turn the world from a *thermostat* into a *dynamical system*. In Lesson 0
> the food refilled to a fixed target and dead ants instantly respawned — both
> food and population were clamped, so nothing could ever really collapse or
> boom. Here we cut both leashes and watch genuine population dynamics emerge:
> stability, boom-bust cycles, and extinction.

This finishes an idea from the original 1996 project — Rule 3 listed
"procreation" as an action, and the essays talked about *survival of the fittest
groups*. None of that can happen if the population is pinned. Now it isn't.

## What changed

**Food is now a renewable resource with its own dynamics.** Each cell updates by
discrete logistic growth, plus diffusion, plus a little spore rain:

```
F  ←  F + r·F·(1 − F/K)  +  D·(neighbour_mean − F)  +  spores
```

- **r** — the intrinsic growth rate. This is *the* control knob, the direct
  analogue of `r` in the logistic map. Small r → food barely regrows; large r →
  food explodes and overshoots.
- **K** — per-cell carrying capacity (`max_food_size`).
- **D** — diffusion. Needed because pure logistic growth from `F = 0` stays 0
  forever; food has to be able to *spread* into empty cells to recolonize.
- **spores** — rare random reseeding so a fully grazed-out field can recover.

**Population is now free.** Dead ants are *not* respawned — they're gone. And a
well-fed ant can choose a new 7th action, **REPRODUCE**, splitting its energy
with an offspring placed in a nearby empty slot. So the number of ants rises and
falls on its own, limited by how much food the world can grow.

> Implementation note: variable population stays GPU-friendly via a **slot pool**
> — we allocate `max_ants` columns and carry an `alive` mask. Births fill dead
> slots; deaths clear the mask; dead slots are forced to do nothing and filtered
> from the view. "No free slot" is a natural ceiling. Reproduction assigns
> offspring to free slots with a fully vectorized cumsum-rank trick (no Python
> loop over worlds). See `env.py:_reproduce`.

## The regimes — drive them with the `r` slider

With the **Ecosystem** preset, slide **Food growth rate r** and watch the
population sparkline:

- **r too low** → food can't keep pace with grazing → the population **crashes to
  extinction**. (Your "too few, goes to zero.")
- **moderate r** → ants and food settle into a **stable coexistence**. (Your
  "about right, stabilizes.")
- **high r** → over-enrichment **destabilizes** into large **boom-bust cycles**:
  food blooms → ants overbreed → overgraze → food busts → ants starve → food
  recovers → repeat. (Your "too much, wildly tips between extremes.")

That last one has a famous name: the **paradox of enrichment** — making the world
*richer* makes it *less* stable. It's a real result in theoretical ecology
(Rosenzweig, 1971), and here it falls out of ants foraging.

## The bifurcation diagram — the architecture paying off

Click **▶ run sweep**. This launches the parallel-worlds trick we built for:
**~200 simulated worlds at once, each with a different `r`**, run past the
transient, sampling each world's population over a window. The result is plotted
as population vs `r`:

- a **thin band** = a stable fixed point (population settles to one level),
- a band that **widens / splits** = oscillation (boom-bust limit cycle) heading
  toward chaos,
- **empty at the left** = the extinction zone.

The whole sweep runs in a few seconds on the GPU. This is the literal point of
the batch dimension from Lesson 0: a parameter sweep is just "many worlds at
once," and a bifurcation diagram is the most satisfying way to *see* it.

> A note on honesty: this is a stochastic, spatial agent-based model, not a clean
> 1-D map, so the diagram is fuzzier than a textbook logistic-map cascade. The
> *regimes* (extinction / stable / oscillating) are robust and clearly visible;
> crisp period-doubling forks are blurred by spatial noise. That's the real
> texture of dynamics in a simulated world, and worth seeing as such.

## Why this matters for the RL lessons ahead

In a clamped world, "survival" was fake — nobody truly starved, population was
fixed, so there was no real selection pressure behind cooperation. Now there is.
When food is genuinely scarce and reproduction is real, *group* behaviors like
sharing where food is can change who survives and reproduces. The altruism
question we're ultimately chasing (Lessons 3–4) becomes a live evolutionary
pressure instead of a hand-tuned bonus. The ecosystem isn't a detour from the RL
goal — it's the stage that makes the goal meaningful.

## Things to try

- Park `r` just past the extinction edge and watch the population teeter.
- Crank `r` to the top and watch the boom-bust sawtooth in the population chart.
- Drop **food diffusion** to 0 — food can no longer recolonize grazed patches;
  watch fragmentation and local extinctions.
- Raise **birth threshold** — ants reproduce more cautiously; the booms soften.
