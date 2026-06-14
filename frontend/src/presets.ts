import type { SimConfig } from "./types";

// Two worlds, two parameter regimes. Toggling mode in the UI sends a reset with
// the matching preset. These eco numbers are the ones tuned to sit near the
// interesting part of the bifurcation (extinction -> stable -> boom-bust).

export const HOMEOSTATIC_PRESET: Partial<SimConfig> = {
  ecosystem: false,
  food_model: "logistic",
  n_ants: 120,
  world_size: 48,
  energy_cost: 0.3,
  bite_size: 1.0,
  max_food_size: 60,
  food_density: 0.04,
};

export const ECOSYSTEM_PRESET: Partial<SimConfig> = {
  ecosystem: true,
  food_model: "logistic",
  n_ants: 250,
  max_ants: 6000,
  world_size: 48,
  energy_cost: 0.5,
  bite_size: 3.0,
  max_food_size: 12,
  food_density: 0.06,      // sparse initial PATCHES, not a carpet
  food_growth_rate: 1.2,   // the bifurcation knob; slide it for boom-bust / extinction
  food_diffusion: 0.0,     // OFF on purpose: diffusion floods the whole grid
  food_seed: 0.06,         // spore rain reseeds new patches so food stays patchy + scarce
  birth_threshold: 0.6,
  birth_cost: 0.5,
};

// Lesson 0.6: food grows from a conserved nutrient pool. No hard food cap; a
// zeroed cell germinates back from the nutrient; mass cycles (closed when
// nutrient_inflow=0). Sustains a population at high r where logistic goes extinct.
export const NUTRIENT_PRESET: Partial<SimConfig> = {
  ecosystem: true,
  food_model: "nutrient",
  n_ants: 200,
  max_ants: 4000,
  world_size: 48,
  energy_cost: 0.5,
  bite_size: 3.0,
  food_growth_rate: 1.8,   // try pushing this high — it cycles instead of collapsing
  birth_threshold: 0.6,
  birth_cost: 0.5,
  nutrient_init: 16,       // total mass budget per cell (closed-system carrying capacity)
  germination: 0.06,       // stochastic N->food sprouting (mass-conserving "spore rain")
  half_sat: 3.0,           // Monod half-saturation for nutrient-limited growth
  nutrient_inflow: 0.0,    // 0 = closed (mass conserved); raise for an open "sunlit" world
  food_diffusion: 0.0,     // nutrient diffusion (spreads the soil); 0 keeps patches sharp
};
