import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev server runs on 5173 and talks to the FastAPI backend on 8000 (see
// src/useSim.ts). `npm run build` emits ./dist, which the backend serves
// directly in production.
export default defineConfig({
  plugins: [react()],
  build: { outDir: "dist" },
});
