import type { LearnerTelemetry } from "./types";

// The learner's mind, made visible. The star is the Q-table heatmap: one row per
// state, one column per action, colour = how good the learner currently thinks
// that action is in that state (green = good, red = bad/death). Watch it fill in
// and differentiate as it learns; the ringed cell in each row is the action it
// would pick (its current policy).

const ACT_ABBR: Record<string, string> = {
  eat: "eat", broadcast: "cast", nothing: "nop", migrate: "migr",
  listen: "lstn", randmove: "move", reproduce: "rep",
};

function cellColor(v: number, scale: number): string {
  const t = Math.max(-1, Math.min(1, v / scale));
  if (t >= 0) return `rgba(70,255,140,${0.08 + 0.85 * t})`; // good → green
  return `rgba(255,80,95,${0.08 + 0.85 * -t})`; // bad → red
}

function Spark({ data, color }: { data: number[]; color: string }) {
  const w = 100, h = 22;
  if (data.length < 2) return <svg width={w} height={h} />;
  const min = Math.min(...data), max = Math.max(...data), span = max - min || 1;
  const d = data
    .map((v, i) => `${i === 0 ? "M" : "L"}${((i / (data.length - 1)) * w).toFixed(1)},${(h - 2 - ((v - min) / span) * (h - 4)).toFixed(1)}`)
    .join(" ");
  return <svg width={w} height={h}><path d={d} fill="none" stroke={color} strokeWidth={1.4} /></svg>;
}

interface Props {
  learner: LearnerTelemetry;
  epsHistory: number[];
  rewardHistory: number[];
  onToggleLearning: () => void;
  onReset: () => void;
}

export function LearnerPanel({ learner, epsHistory, rewardHistory, onToggleLearning, onReset }: Props) {
  const { q_table, states, actions, epsilon, reward, learning } = learner;
  const scale = Math.max(1e-6, ...q_table.flat().map((v) => Math.abs(v)));

  return (
    <div className="learner">
      <div className="bif-head">
        <div className="group-title" style={{ margin: 0 }}>Brain — tabular Q-learning</div>
        <div className="row" style={{ gap: 8 }}>
          <button className={"seg-btn" + (learning ? " active" : "")} onClick={onToggleLearning}>
            {learning ? "● learning" : "❚❚ frozen"}
          </button>
          <button className="btn" onClick={onReset} style={{ flex: "0 0 auto", padding: "6px 10px" }}>
            ↺ reset brain
          </button>
        </div>
      </div>

      <div className="learner-stats">
        <div className="lstat">
          <span>exploration ε <b style={{ color: "#c77dff" }}>{epsilon.toFixed(3)}</b></span>
          <Spark data={epsHistory} color="#c77dff" />
        </div>
        <div className="lstat">
          <span>mean reward <b style={{ color: reward >= 0 ? "#7cffcb" : "#ff5577" }}>{reward.toFixed(2)}</b></span>
          <Spark data={rewardHistory} color="#7cffcb" />
        </div>
      </div>

      <div className="qtable" style={{ gridTemplateColumns: `74px repeat(${actions.length}, 1fr)` }}>
        <div className="qcorner" />
        {actions.map((a) => <div key={a} className="qhead" title={a}>{ACT_ABBR[a] ?? a}</div>)}
        {q_table.map((row, s) => {
          const best = row.indexOf(Math.max(...row));
          return [
            <div className="qrowlabel" key={`l${s}`} title={states[s]}>{states[s]}</div>,
            ...row.map((v, a) => (
              <div key={`${s}-${a}`}
                   className={"qcell" + (a === best ? " qbest" : "")}
                   style={{ background: cellColor(v, scale) }}
                   title={`${states[s]} · ${actions[a]} = ${v.toFixed(3)}`} />
            )),
          ];
        })}
      </div>
      <div className="hint">
        Rows = states (energy bin · on food? · heard a tip?), columns = actions. Green = the
        learner expects a good outcome, red = bad (e.g. the death penalty). The ringed cell is
        what it would do. Freeze it to watch the learned policy without further exploration.
      </div>
    </div>
  );
}
