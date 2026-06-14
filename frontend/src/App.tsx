import { useEffect, useMemo, useState } from "react";
import { useSim } from "./useSim";
import { AntCanvas } from "./AntCanvas";
import { Controls } from "./Controls";
import { MetricsPanel } from "./Metrics";
import type { SimConfig } from "./types";

export default function App() {
  const { frame, connected, history, send } = useSim();
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(3);

  const config = frame?.config ?? null;
  const policy = frame?.policy ?? "heuristic";
  const snapshot = frame?.snapshot ?? null;

  // Push the initial speed once we're connected.
  useEffect(() => {
    if (connected) send({ type: "speed", steps_per_frame: speed, fps: 30 });
  }, [connected]);

  const startPause = () => {
    const next = !running;
    setRunning(next);
    send({ type: next ? "start" : "pause" });
  };

  const reset = () => {
    send({ type: "reset", config });
  };

  const onConfig = (updates: Partial<SimConfig>, structural: boolean) => {
    if (structural) {
      send({ type: "reset", config: { ...config, ...updates } });
    } else {
      send({ type: "config", config: updates });
    }
  };

  const onSpeed = (n: number) => {
    setSpeed(n);
    send({ type: "speed", steps_per_frame: n, fps: 30 });
  };

  const step = snapshot?.metrics.step ?? 0;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">🐜</span>
          <div>
            <h1>Communicating Ants</h1>
            <div className="subtitle">Lesson 0 — The World · a vectorized, GPU-resident revival</div>
          </div>
        </div>
        <div className={"status " + (connected ? "ok" : "bad")}>
          {connected ? `● live · step ${step.toLocaleString()}` : "○ disconnected"}
        </div>
      </header>

      <main className="layout">
        <Controls
          config={config}
          running={running}
          policy={policy}
          speed={speed}
          onStartPause={startPause}
          onReset={reset}
          onPolicy={(name) => send({ type: "policy", name })}
          onSpeed={onSpeed}
          onConfig={onConfig}
        />

        <section className="stage">
          <AntCanvas snapshot={snapshot} />
          <Legend />
        </section>

        <MetricsPanel history={history} />
      </main>
    </div>
  );
}

function Legend() {
  const items: [string, string][] = [
    ["#ffe600", "eat / food"],
    ["#ff3344", "broadcast"],
    ["#ffffff", "listen"],
    ["#ff45e0", "teleport"],
    ["#c77dff", "move"],
    ["#3b6bff", "rest"],
    ["#3cff8c", "comm link"],
  ];
  return (
    <div className="legend">
      {items.map(([c, label]) => (
        <span className="legend-item" key={label}>
          <span className="dot" style={{ background: c }} /> {label}
        </span>
      ))}
    </div>
  );
}
