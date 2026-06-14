import { useRef, useState } from "react";
import type { BifurcationData } from "./types";

// Fetches the parallel-worlds sweep over food growth rate r and draws the
// bifurcation diagram: each column is one r, each dot a sampled population level
// after the transient. A thin vertical band = a stable population; a wide /
// split band = boom-bust oscillation or chaos (the paradox of enrichment).
function apiBase(): string {
  return import.meta.env.DEV ? `http://${location.hostname}:8000` : "";
}

function draw(canvas: HTMLCanvasElement, d: BifurcationData) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const W = canvas.width, H = canvas.height;
  const padL = 38, padB = 26, padT = 10, padR = 10;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const rMin = d.r[0], rMax = d.r[d.r.length - 1];
  const x = (r: number) => padL + ((r - rMin) / (rMax - rMin)) * plotW;
  const y = (v: number) => padT + (1 - v) * plotH; // v is pop fraction 0..1

  ctx.fillStyle = "#05060f";
  ctx.fillRect(0, 0, W, H);

  ctx.strokeStyle = "rgba(140,150,200,0.25)";
  ctx.fillStyle = "rgba(170,180,220,0.8)";
  ctx.font = "10px ui-monospace, monospace";
  ctx.beginPath();
  ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + plotH); ctx.lineTo(padL + plotW, padT + plotH);
  ctx.stroke();
  ctx.fillText("pop", 6, padT + 8);
  ctx.fillText(`r = ${rMin.toFixed(1)}`, padL, H - 8);
  ctx.fillText(`${rMax.toFixed(1)}`, padL + plotW - 18, H - 8);

  ctx.fillStyle = "rgba(124,255,203,0.5)";
  for (const [r, v] of d.points) ctx.fillRect(x(r) - 0.6, y(v) - 0.6, 1.2, 1.2);

  ctx.strokeStyle = "rgba(255,210,90,0.9)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  d.r.forEach((r, i) => {
    const px = x(r), py = y(d.mean[i]);
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  });
  ctx.stroke();
}

function NumInput({ label, value, set, min, max, step }: {
  label: string; value: number; set: (n: number) => void;
  min: number; max: number; step: number;
}) {
  return (
    <label className="bif-num">
      <span>{label}</span>
      <input type="number" min={min} max={max} step={step} value={value}
             onChange={(e) => set(Math.max(min, Math.min(max, Number(e.target.value) || min)))} />
    </label>
  );
}

export function Bifurcation() {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  // Sweep controls. "Settle" steps let each world reach its attractor before we
  // start sampling; if oscillations don't look settled, raise it. "Sample" is
  // the window we record (longer = a slow cycle's full swing is captured).
  const [settle, setSettle] = useState(400);
  const [sample, setSample] = useState(240);
  const [nr, setNr] = useState(200);
  const [rMin, setRMin] = useState(0.3);
  const [rMax, setRMax] = useState(3.0);
  const [runs, setRuns] = useState(1);

  const run = async () => {
    setLoading(true); setErr(null); setInfo(null);
    try {
      const t0 = performance.now();
      const lo = Math.min(rMin, rMax - 0.01);
      const q = `n_r=${nr}&transient=${settle}&sample=${sample}&runs_per_r=${runs}` +
        `&r_min=${lo}&r_max=${rMax}`;
      const res = await fetch(`${apiBase()}/api/bifurcation?${q}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const d = (await res.json()) as BifurcationData;
      if (ref.current) draw(ref.current, d);
      const secs = ((performance.now() - t0) / 1000).toFixed(1);
      const replicates = d.runs_per_r > 1 ? ` × ${d.runs_per_r} runs` : "";
      setInfo(`${d.r.length} r-values${replicates} · ${settle}+${sample} steps · y=1.0 ≈ ${Math.round(d.pop_ref)} ants · ${secs}s on GPU`);
    } catch (e: any) {
      setErr(e.message ?? "failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bifurcation">
      <div className="bif-head">
        <div className="group-title" style={{ margin: 0 }}>Bifurcation diagram</div>
        <button className="btn" onClick={run} disabled={loading}>
          {loading ? "running sweep…" : "▶ run sweep"}
        </button>
      </div>
      <div className="bif-controls">
        <NumInput label="r min" value={rMin} set={setRMin} min={0} max={4} step={0.05} />
        <NumInput label="r max" value={rMax} set={setRMax} min={0.05} max={4} step={0.05} />
        <NumInput label="r values" value={nr} set={setNr} min={10} max={512} step={10} />
        <NumInput label="runs / r" value={runs} set={setRuns} min={1} max={12} step={1} />
        <NumInput label="settle" value={settle} set={setSettle} min={100} max={3000} step={100} />
        <NumInput label="sample" value={sample} set={setSample} min={50} max={1500} step={50} />
      </div>
      <canvas ref={ref} width={540} height={240} className="bif-canvas" />
      <div className="hint">
        Each column is one parallel world with a different food growth rate <b>r</b>.
        Thin band → stable. Wide / split band → boom-bust oscillation &amp; chaos.
        Far left → extinction. <b>Zoom</b> with r min/max and raise <b>runs/r</b>
        (replicates at the same r, different noise) to test whether a low-r dropout is
        a real island of stability or just one unlucky run; raise <b>settle</b>/<b>sample</b>
        for slow oscillations. The live view pauses for the few seconds the sweep runs.
        {info && <span style={{ color: "#7cffcb" }}> · {info}</span>}
        {err && <span style={{ color: "#ff6b81" }}> · {err}</span>}
      </div>
    </div>
  );
}
