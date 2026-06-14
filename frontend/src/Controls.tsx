import { useEffect, useRef, useState } from "react";
import type { SimConfig, WorldModel } from "./types";

// The reborn 1997 slider panel, now with three world models. Live (physics) knobs
// take effect next step; structural knobs (ant count, world size, mode) rebuild.
const STRUCTURAL_FIELDS: (keyof SimConfig)[] = ["n_ants", "world_size", "max_ants"];

const ALL: WorldModel[] = ["homeostatic", "ecosystem", "nutrient"];
const ECO: WorldModel[] = ["ecosystem", "nutrient"];        // free-population modes
const LOGI: WorldModel[] = ["homeostatic", "ecosystem"];    // hard per-cell food cap

interface Spec { key: keyof SimConfig; label: string; min: number; max: number; step: number; modes: WorldModel[]; }
const SPECS: Spec[] = [
  { key: "n_ants", label: "Number of ants", min: 5, max: 600, step: 1, modes: ALL },
  { key: "world_size", label: "World size", min: 12, max: 120, step: 1, modes: ALL },
  { key: "food_growth_rate", label: "Food growth rate  r", min: 0.2, max: 4.0, step: 0.05, modes: ECO },
  { key: "food_seed", label: "Spore rain rate", min: 0, max: 0.3, step: 0.01, modes: ["ecosystem"] },
  { key: "germination", label: "Germination rate", min: 0, max: 0.3, step: 0.01, modes: ["nutrient"] },
  { key: "half_sat", label: "Nutrient half-sat  h", min: 0.5, max: 20, step: 0.5, modes: ["nutrient"] },
  { key: "nutrient_init", label: "Nutrient / cell", min: 1, max: 60, step: 1, modes: ["nutrient"] },
  { key: "nutrient_inflow", label: "Inflow (source)", min: 0, max: 2, step: 0.02, modes: ["nutrient"] },
  { key: "nutrient_loss", label: "Washout (sink)", min: 0, max: 0.1, step: 0.005, modes: ["nutrient"] },
  { key: "food_diffusion", label: "Diffusion  D", min: 0, max: 0.25, step: 0.005, modes: ECO },
  { key: "birth_threshold", label: "Birth threshold", min: 0.3, max: 1.0, step: 0.05, modes: ECO },
  { key: "birth_cost", label: "Birth cost", min: 0.1, max: 0.8, step: 0.05, modes: ECO },
  { key: "max_ants", label: "Population cap", min: 500, max: 6000, step: 250, modes: ECO },
  { key: "energy_max", label: "Energy capacity", min: 5, max: 80, step: 1, modes: ALL },
  { key: "energy_cost", label: "Energy cost / step", min: 0.05, max: 1.5, step: 0.05, modes: ALL },
  { key: "food_density", label: "Food density / patch frac", min: 0.005, max: 0.2, step: 0.005, modes: LOGI },
  { key: "max_food_size", label: "Max food pile / cell K", min: 5, max: 150, step: 1, modes: LOGI },
  { key: "bite_size", label: "Bite size", min: 0.25, max: 5, step: 0.25, modes: ALL },
  { key: "move_radius", label: "Move radius", min: 0.5, max: 8, step: 0.5, modes: ALL },
  { key: "comm_radius", label: "Comm radius", min: 1, max: 30, step: 1, modes: ALL },
];

const MODE_HINT: Record<WorldModel, string> = {
  homeostatic: "Food refills to target; dead ants respawn. Population pinned (Lesson 0).",
  ecosystem: "Renewable logistic food + free population: births, deaths, boom-bust (Lesson 0.5).",
  nutrient: "Food grows from a conserved nutrient pool — no hard cap, zeroed cells regrow, mass cycles (Lesson 0.6).",
};

interface Props {
  config: SimConfig | null;
  mode: WorldModel;
  running: boolean;
  policy: string;
  speed: number;
  onStartPause: () => void;
  onReset: () => void;
  onMode: (model: WorldModel) => void;
  onPolicy: (name: string) => void;
  onSpeed: (n: number) => void;
  onConfig: (updates: Partial<SimConfig>, structural: boolean) => void;
}

export function Controls(p: Props) {
  const [local, setLocal] = useState<SimConfig | null>(p.config);

  // Resync local sliders from the server whenever the run's *structure* changes
  // (reset, mode switch, reconnect, or a structural reset from another tab), so
  // sliders can't drift out of sync with the actual simulation.
  const sig = p.config
    ? `${p.config.ecosystem}|${p.config.food_model}|${p.config.world_size}|${p.config.n_ants}|${p.config.max_ants}`
    : "";
  useEffect(() => { if (p.config) setLocal(p.config); }, [sig]);
  useEffect(() => { if (p.config && !local) setLocal(p.config); }, [p.config]);

  const cfg = local ?? p.config;
  // Debounce live knobs (sliders fire many onChange events); apply structural
  // knobs only on release so dragging doesn't trigger a storm of world rebuilds.
  const liveTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  if (!cfg) return <aside className="panel">connecting…</aside>;

  const isStructural = (key: keyof SimConfig) => STRUCTURAL_FIELDS.includes(key);

  const onSlide = (key: keyof SimConfig, value: number) => {
    setLocal({ ...cfg, [key]: value } as SimConfig);     // instant visual feedback
    if (!isStructural(key)) {
      clearTimeout(liveTimer.current);
      liveTimer.current = setTimeout(
        () => p.onConfig({ [key]: value } as Partial<SimConfig>, false), 80);
    }
  };

  const onCommit = (key: keyof SimConfig, value: number) => {
    if (isStructural(key)) p.onConfig({ [key]: value } as Partial<SimConfig>, true);
  };

  const specs = SPECS.filter((s) => s.modes.includes(p.mode));

  return (
    <aside className="panel">
      <div className="row buttons">
        <button className={p.running ? "btn pause" : "btn start"} onClick={p.onStartPause}>
          {p.running ? "❚❚ Pause" : "▶ Start"}
        </button>
        <button className="btn" onClick={p.onReset}>↺ Reset</button>
      </div>

      <div className="group">
        <div className="group-title">World model</div>
        <div className="row seg">
          {ALL.map((m) => (
            <button key={m} className={"seg-btn" + (p.mode === m ? " active" : "")}
                    onClick={() => p.onMode(m)}>
              {m === "homeostatic" ? "Homeo." : m === "ecosystem" ? "Eco." : "Nutrient"}
            </button>
          ))}
        </div>
        <div className="hint">{MODE_HINT[p.mode]}</div>
      </div>

      <div className="group">
        <div className="group-title">Brain</div>
        <div className="row seg">
          {["heuristic", "random"].map((name) => (
            <button key={name} className={"seg-btn" + (p.policy === name ? " active" : "")}
                    onClick={() => p.onPolicy(name)}>{name}</button>
          ))}
        </div>
      </div>

      <div className="group">
        <div className="group-title">Speed — {p.speed} steps / frame</div>
        <input type="range" min={1} max={50} step={1} value={p.speed}
               onChange={(e) => p.onSpeed(Number(e.target.value))} />
      </div>

      <div className="group">
        <div className="group-title">{p.mode === "homeostatic" ? "World" : "Ecosystem & world"}</div>
        {specs.map((s) => (
          <label className="slider" key={s.key}>
            <span className="slabel">{s.label}</span>
            <span className="sval">{fmt(cfg[s.key] as number)}</span>
            <input type="range" min={s.min} max={s.max} step={s.step}
                   value={cfg[s.key] as number}
                   onChange={(e) => onSlide(s.key, Number(e.target.value))}
                   onPointerUp={(e) => onCommit(s.key, Number(e.currentTarget.value))}
                   onKeyUp={(e) => onCommit(s.key, Number(e.currentTarget.value))} />
          </label>
        ))}
        <div className="hint">Ant count &amp; world size restart the world; the rest apply live.</div>
      </div>
    </aside>
  );
}

function fmt(v: number) {
  return Number.isInteger(v) ? String(v) : v.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}
