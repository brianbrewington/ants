// Shapes mirror what the backend streams (see antfarm/env.py:snapshot and
// antfarm/server.py). Kept in one place so the renderer and controls agree.

export interface Metrics {
  step: number;
  food_eaten: number;
  deaths: number;
  mean_energy: number;
  total_food: number;
  frac_eat: number;
  frac_broadcast: number;
  frac_listen: number;
  frac_move: number;
}

export interface Snapshot {
  world_size: number;
  ants: { pos: [number, number][]; energy: number[]; action: number[] };
  food: [number, number, number][]; // x, y, amount
  links: [[number, number], [number, number]][];
  metrics: Metrics;
  energy_max: number;
}

export interface SimConfig {
  world_size: number;
  n_ants: number;
  n_worlds: number;
  energy_max: number;
  energy_cost: number;
  food_density: number;
  max_food_size: number;
  bite_size: number;
  move_radius: number;
  comm_radius: number;
  seed: number;
  device: string;
}

export interface Frame {
  type: "frame";
  snapshot: Snapshot;
  policy: string;
  config: SimConfig;
}

// Action ids must match antfarm/env.py
export const ACTION = {
  EAT: 0,
  BROADCAST: 1,
  NOTHING: 2,
  TELEPORT: 3,
  LISTEN: 4,
  RANDMOVE: 5,
} as const;

// Colours echo the 1997 MATLAB legend.
export const ACTION_COLOR: Record<number, string> = {
  0: "#ffe600", // eat     - yellow
  1: "#ff3344", // broadcast - red
  2: "#3b6bff", // nothing - blue
  3: "#ff45e0", // teleport - magenta
  4: "#ffffff", // listen  - white
  5: "#c77dff", // randmove - violet
};
