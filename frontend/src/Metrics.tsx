import type { Metrics } from "./types";
import { ACTION, ACTION_COLOR } from "./types";

// What each action does — shown as a hover tooltip on its bar. Order here is the
// display order in the panel.
const ACTION_META: { name: string; aid: number; tip: string }[] = [
  { name: "eat", aid: ACTION.EAT, tip: "Consume food in the current cell (if any, and not already full)." },
  { name: "broadcast", aid: ACTION.BROADCAST, tip: "Announce your location to nearby listeners." },
  { name: "listen", aid: ACTION.LISTEN, tip: "Lock onto the nearest broadcaster as your destination." },
  { name: "migrate", aid: ACTION.TELEPORT, tip: "Migrate to the location heard from a broadcast — the act-on-information half of communication (a mobile agent hopping to an advertised node)." },
  { name: "move", aid: ACTION.RANDMOVE, tip: "Wander a short random step." },
  { name: "nothing", aid: ACTION.NOTHING, tip: "Rest — do nothing this step." },
  { name: "reproduce", aid: ACTION.REPRODUCE, tip: "Spend energy to place an offspring in a nearby empty slot." },
];

// Lightweight inline-SVG sparkline. The 1997 version plotted four population
// metrics over time; we keep that spirit. As the lessons add learning, this is
// where "is it actually getting better?" will show up.
function Spark({ data, color, label, fmt }: {
  data: number[]; color: string; label: string; fmt: (v: number) => string;
}) {
  const w = 220, h = 44;
  const last = data.length ? data[data.length - 1] : 0;
  let path = "";
  if (data.length > 1) {
    const min = Math.min(...data), max = Math.max(...data);
    const span = max - min || 1;
    path = data
      .map((v, i) => {
        const x = (i / (data.length - 1)) * w;
        const y = h - 4 - ((v - min) / span) * (h - 8);
        return `${i === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  }
  return (
    <div className="spark">
      <div className="spark-head">
        <span className="spark-label" style={{ color }}>{label}</span>
        <span className="spark-val">{fmt(last)}</span>
      </div>
      <svg width={w} height={h}>
        <path d={path} fill="none" stroke={color} strokeWidth={1.5} />
      </svg>
    </div>
  );
}

export function MetricsPanel({ history, ecosystem, mode }:
  { history: Metrics[]; ecosystem: boolean; mode: string }) {
  const col = (k: keyof Metrics) => history.map((m) => m[k] as number);
  const m = history.length ? history[history.length - 1] : null;

  // All 7 actions, always shown. `reproduce` is greyed out in Homeostatic mode
  // (it's a no-op there since the population is fixed) — the tooltip explains why.
  const fracOf: Record<number, number> = m
    ? { 0: m.frac_eat, 1: m.frac_broadcast, 2: m.frac_nothing, 3: m.frac_teleport,
        4: m.frac_listen, 5: m.frac_move, 6: m.frac_reproduce }
    : {};
  const actions = (m ? ACTION_META : []).map((a) => ({
    ...a,
    frac: fracOf[a.aid] ?? 0,
    active: a.aid !== ACTION.REPRODUCE || ecosystem,
  }));

  return (
    <div className="metrics">
      <div className="sparks">
        {ecosystem && (
          <Spark data={col("population")} color="#9b8cff" label="Population (alive)"
                 fmt={(v) => v.toFixed(0)} />
        )}
        <Spark data={col("mean_energy")} color="#7CFFCB" label="Mean energy"
               fmt={(v) => v.toFixed(1)} />
        {ecosystem && (
          <Spark data={col("births")} color="#46ff7a" label="Births / frame"
                 fmt={(v) => v.toFixed(0)} />
        )}
        <Spark data={col("deaths")} color="#ff5577" label="Deaths / frame"
               fmt={(v) => v.toFixed(0)} />
        <Spark data={col("total_food")} color="#ffb347" label="Food on grid"
               fmt={(v) => v.toFixed(0)} />
        {mode === "nutrient" && (
          <Spark data={col("total_nutrient")} color="#8a6f4e" label="Nutrient in soil"
                 fmt={(v) => v.toFixed(0)} />
        )}
      </div>

      <div className="actionbars">
        <div className="group-title">Action mix</div>
        {actions.map(({ name, frac, aid, tip, active }) => {
          const color = active ? ACTION_COLOR[aid] : "#5a6080";
          const title = active
            ? tip
            : `${tip}\n\n(Greyed out: only active in Ecosystem mode — the population is fixed in Homeostatic mode, so reproduction does nothing.)`;
          return (
            <div className={"abar" + (active ? "" : " inactive")} key={name} title={title}>
              <span className="abar-label" style={{ color }}>{name}</span>
              <div className="abar-track">
                <div className="abar-fill"
                     style={{ width: `${(frac * 100).toFixed(0)}%`, background: color }} />
              </div>
              <span className="abar-val">{(frac * 100).toFixed(0)}%</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
