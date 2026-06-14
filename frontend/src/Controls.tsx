import { useEffect, useState } from "react";
import type { SimConfig } from "./types";

// The reborn 1997 slider panel. Live (physics) knobs take effect next step;
// structural knobs (ant count, world size) rebuild the world via a reset.
const LIVE_FIELDS: (keyof SimConfig)[] = [
  "energy_cost", "food_density", "max_food_size", "bite_size",
  "move_radius", "comm_radius", "energy_max",
];
const STRUCTURAL_FIELDS: (keyof SimConfig)[] = ["n_ants", "world_size"];

interface Spec { key: keyof SimConfig; label: string; min: number; max: number; step: number; }
const SPECS: Spec[] = [
  { key: "n_ants", label: "Number of ants", min: 5, max: 600, step: 1 },
  { key: "world_size", label: "World size", min: 12, max: 120, step: 1 },
  { key: "energy_max", label: "Energy capacity", min: 5, max: 80, step: 1 },
  { key: "energy_cost", label: "Energy cost / step", min: 0.05, max: 1.5, step: 0.05 },
  { key: "food_density", label: "Food density", min: 0.005, max: 0.2, step: 0.005 },
  { key: "max_food_size", label: "Max food pile", min: 5, max: 150, step: 1 },
  { key: "bite_size", label: "Bite size", min: 0.25, max: 5, step: 0.25 },
  { key: "move_radius", label: "Move radius", min: 0.5, max: 8, step: 0.5 },
  { key: "comm_radius", label: "Comm radius", min: 1, max: 30, step: 1 },
];

interface Props {
  config: SimConfig | null;
  running: boolean;
  policy: string;
  speed: number;
  onStartPause: () => void;
  onReset: () => void;
  onPolicy: (name: string) => void;
  onSpeed: (n: number) => void;
  onConfig: (updates: Partial<SimConfig>, structural: boolean) => void;
}

export function Controls(p: Props) {
  // Local mirror so sliders feel snappy; synced when a fresh config arrives.
  const [local, setLocal] = useState<SimConfig | null>(p.config);
  useEffect(() => { if (p.config && !local) setLocal(p.config); }, [p.config]);
  const cfg = local ?? p.config;
  if (!cfg) return <aside className="panel">connecting…</aside>;

  const change = (key: keyof SimConfig, value: number) => {
    const next = { ...cfg, [key]: value } as SimConfig;
    setLocal(next);
    const structural = STRUCTURAL_FIELDS.includes(key);
    p.onConfig({ [key]: value } as Partial<SimConfig>, structural);
  };

  return (
    <aside className="panel">
      <div className="row buttons">
        <button className={p.running ? "btn pause" : "btn start"} onClick={p.onStartPause}>
          {p.running ? "❚❚ Pause" : "▶ Start"}
        </button>
        <button className="btn" onClick={p.onReset}>↺ Reset</button>
      </div>

      <div className="group">
        <div className="group-title">Brain</div>
        <div className="row seg">
          {["heuristic", "random"].map((name) => (
            <button
              key={name}
              className={"seg-btn" + (p.policy === name ? " active" : "")}
              onClick={() => p.onPolicy(name)}
            >
              {name}
            </button>
          ))}
        </div>
        <div className="hint">Lesson 0 has no learning yet — these are hand-written baselines.</div>
      </div>

      <div className="group">
        <div className="group-title">Speed — {p.speed} steps / frame</div>
        <input type="range" min={1} max={50} step={1} value={p.speed}
               onChange={(e) => p.onSpeed(Number(e.target.value))} />
      </div>

      <div className="group">
        <div className="group-title">World</div>
        {SPECS.map((s) => (
          <label className="slider" key={s.key}>
            <span className="slabel">{s.label}</span>
            <span className="sval">{fmt(cfg[s.key] as number)}</span>
            <input type="range" min={s.min} max={s.max} step={s.step}
                   value={cfg[s.key] as number}
                   onChange={(e) => change(s.key, Number(e.target.value))} />
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
