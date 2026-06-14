import type { Metrics } from "./types";
import { ACTION_COLOR } from "./types";

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

export function MetricsPanel({ history, ecosystem }: { history: Metrics[]; ecosystem: boolean }) {
  const col = (k: keyof Metrics) => history.map((m) => m[k] as number);
  const m = history.length ? history[history.length - 1] : null;

  const actions: [string, number, number][] = m
    ? [
        ["eat", m.frac_eat, 0],
        ["broadcast", m.frac_broadcast, 1],
        ["listen", m.frac_listen, 4],
        ["move", m.frac_move, 5],
      ]
    : [];

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
      </div>

      <div className="actionbars">
        <div className="group-title">Action mix</div>
        {actions.map(([name, frac, aid]) => (
          <div className="abar" key={name}>
            <span className="abar-label" style={{ color: ACTION_COLOR[aid] }}>{name}</span>
            <div className="abar-track">
              <div className="abar-fill"
                   style={{ width: `${(frac * 100).toFixed(0)}%`, background: ACTION_COLOR[aid] }} />
            </div>
            <span className="abar-val">{(frac * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
