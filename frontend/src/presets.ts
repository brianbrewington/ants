import type { SimConfig } from "./types";

// Two worlds, two parameter regimes. Toggling mode in the UI sends a reset with
// the matching preset. These eco numbers are the ones tuned to sit near the
// interesting part of the bifurcation (extinction -> stable -> boom-bust).

export const HOMEOSTATIC_PRESET: Partial<SimConfig> = {
  ecosystem: false,
  n_ants: 120,
  world_size: 48,
  energy_cost: 0.3,
  bite_size: 1.0,
  max_food_size: 60,
  food_density: 0.04,
};

export const ECOSYSTEM_PRESET: Partial<SimConfig> = {
  ecosystem: true,
  n_ants: 300,
  max_ants: 6000,
  world_size: 48,
  energy_cost: 0.5,
  bite_size: 3.0,
  max_food_size: 10,
  food_density: 0.12,
  food_growth_rate: 1.0,   // near the onset of oscillation; try sliding it up
  food_diffusion: 0.015,
  food_seed: 0.0,
  birth_threshold: 0.6,
  birth_cost: 0.5,
};
