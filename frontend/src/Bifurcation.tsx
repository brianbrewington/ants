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

  // axes
  ctx.strokeStyle = "rgba(140,150,200,0.25)";
  ctx.fillStyle = "rgba(170,180,220,0.8)";
  ctx.font = "10px ui-monospace, monospace";
  ctx.beginPath();
  ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + plotH); ctx.lineTo(padL + plotW, padT + plotH);
  ctx.stroke();
  ctx.fillText("pop", 6, padT + 8);
  ctx.fillText(`r = ${rMin.toFixed(1)}`, padL, H - 8);
  ctx.fillText(`${rMax.toFixed(1)}`, padL + plotW - 18, H - 8);

  // attractor dots
  ctx.fillStyle = "rgba(124,255,203,0.5)";
  for (const [r, v] of d.points) {
    ctx.fillRect(x(r) - 0.6, y(v) - 0.6, 1.2, 1.2);
  }
  // mean line
  ctx.strokeStyle = "rgba(255,210,90,0.9)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  d.r.forEach((r, i) => {
    const px = x(r), py = y(d.mean[i]);
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py);
  });
  ctx.stroke();
}

export function Bifurcation() {
  const ref = useRef<HTMLCanvasElement | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);

  const run = async () => {
    setLoading(true); setErr(null); setInfo(null);
    try {
      const t0 = performance.now();
      const res = await fetch(`${apiBase()}/api/bifurcation?n_r=200&transient=400&sample=240`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const d = (await res.json()) as BifurcationData;
      if (ref.current) draw(ref.current, d);
      setInfo(`${d.r.length} worlds · y=1.0 ≈ ${Math.round(d.pop_ref)} ants · ${((performance.now() - t0) / 1000).toFixed(1)}s on GPU`);
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
      <canvas ref={ref} width={540} height={240} className="bif-canvas" />
      <div className="hint">
        Each column is one parallel world with a different food growth rate <b>r</b>.
        Thin band → stable population. Wide / split band → boom-bust oscillation &amp;
        chaos. Far left → extinction. {info && <span style={{ color: "#7cffcb" }}>· {info}</span>}
        {err && <span style={{ color: "#ff6b81" }}>· {err}</span>}
      </div>
    </div>
  );
}
