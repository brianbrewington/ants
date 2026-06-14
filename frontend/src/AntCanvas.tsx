import { useEffect, useRef } from "react";
import type { Snapshot } from "./types";
import { ACTION_COLOR } from "./types";

// Renders world 0: food, ants (coloured by their last action), and the green
// communication links between listeners and broadcasters. Pure canvas2d -- fast
// enough for a few hundred ants at 30fps; we can switch to WebGL if we ever
// want to draw thousands.
export function AntCanvas({ snapshot }: { snapshot: Snapshot | null }) {
  const ref = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !snapshot) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = snapshot.world_size;
    const px = canvas.width;
    const s = px / W; // grid units -> pixels

    // Background.
    ctx.fillStyle = "#05060f";
    ctx.fillRect(0, 0, px, px);

    // Faint grid.
    ctx.strokeStyle = "rgba(80,90,140,0.08)";
    ctx.lineWidth = 1;
    const gridStep = Math.max(1, Math.round(W / 16));
    for (let i = 0; i <= W; i += gridStep) {
      ctx.beginPath();
      ctx.moveTo(i * s, 0); ctx.lineTo(i * s, px);
      ctx.moveTo(0, i * s); ctx.lineTo(px, i * s);
      ctx.stroke();
    }

    // Food: amber squares, brighter/bigger with more food.
    for (const [fx, fy, amt] of snapshot.food) {
      const a = Math.min(1, amt / 60);
      const size = s * (0.4 + 0.6 * a);
      ctx.fillStyle = `rgba(255,210,60,${0.25 + 0.6 * a})`;
      ctx.fillRect((fx + 0.5) * s - size / 2, (fy + 0.5) * s - size / 2, size, size);
    }

    // Communication links: glowing green threads.
    ctx.strokeStyle = "rgba(60,255,140,0.55)";
    ctx.lineWidth = 1.25;
    ctx.setLineDash([3, 3]);
    for (const [[lx, ly], [sx, sy]] of snapshot.links) {
      ctx.beginPath();
      ctx.moveTo(lx * s, ly * s);
      ctx.lineTo(sx * s, sy * s);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Ants: dot coloured by action, sized slightly by energy.
    const { pos, energy, action } = snapshot.ants;
    const emax = snapshot.energy_max || 20;
    for (let i = 0; i < pos.length; i++) {
      const [x, y] = pos[i];
      const e = Math.max(0.15, Math.min(1, energy[i] / emax));
      const r = s * (0.22 + 0.22 * e);
      ctx.beginPath();
      ctx.fillStyle = ACTION_COLOR[action[i]] ?? "#ffffff";
      ctx.globalAlpha = 0.5 + 0.5 * e;
      ctx.arc(x * s, y * s, r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }, [snapshot]);

  return <canvas ref={ref} width={680} height={680} className="world" />;
}
