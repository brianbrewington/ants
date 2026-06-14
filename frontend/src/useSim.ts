import { useEffect, useRef, useState, useCallback } from "react";
import type { Frame, Metrics } from "./types";

// Where the backend lives. In dev (vite on :5173) we point at :8000; in a
// production build the FastAPI server hosts both, so we reuse the page host.
function wsUrl(): string {
  const host = import.meta.env.DEV ? `${location.hostname}:8000` : location.host;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${host}/ws`;
}

const HISTORY = 240; // points kept for the metric charts

export function useSim() {
  const [frame, setFrame] = useState<Frame | null>(null);
  const [connected, setConnected] = useState(false);
  const [history, setHistory] = useState<Metrics[]>([]);
  const ws = useRef<WebSocket | null>(null);
  // Track the last frame so we can clear the chart history when the run restarts
  // (step counter goes backwards) or the world model changes — otherwise we'd
  // plot two different runs as one continuous, misleading series.
  const last = useRef<{ step: number; eco: boolean }>({ step: -1, eco: false });

  useEffect(() => {
    let alive = true;
    let socket: WebSocket;
    let retry: ReturnType<typeof setTimeout>;

    const connect = () => {
      socket = new WebSocket(wsUrl());
      ws.current = socket;
      socket.onopen = () => alive && setConnected(true);
      socket.onclose = () => {
        if (!alive) return;
        setConnected(false);
        retry = setTimeout(connect, 1000); // auto-reconnect on backend restart
      };
      socket.onmessage = (ev) => {
        let data: Frame;
        try {
          data = JSON.parse(ev.data) as Frame;
        } catch {
          return; // ignore malformed frame; keep the connection alive
        }
        if (!data || data.type !== "frame") return;
        setFrame(data);
        const step = data.snapshot.metrics.step;
        const eco = data.snapshot.ecosystem;
        const restarted = step < last.current.step || eco !== last.current.eco;
        last.current = { step, eco };
        setHistory((h) => {
          const base = restarted ? [] : h;
          const next = [...base, data.snapshot.metrics];
          return next.length > HISTORY ? next.slice(-HISTORY) : next;
        });
      };
    };
    connect();
    return () => {
      alive = false;
      clearTimeout(retry);
      socket?.close();
    };
  }, []);

  const send = useCallback((msg: unknown) => {
    const s = ws.current;
    if (s && s.readyState === WebSocket.OPEN) s.send(JSON.stringify(msg));
  }, []);

  return { frame, connected, history, send };
}
