// Ground plane hotspot map: where do close calls cluster?
// Renders event peak locations in real world meters on a canvas.
// (Leaflet/GPS integration is future work for georeferenced cameras;
// sample footage has no coordinates, so the scene's own ground plane
// is the honest canvas.)

import { useEffect, useRef, useState } from "react";
import type { HeatPoint } from "../api";

const H = 380;
const PAD = 40;

function scoreColor(score: number, alpha = 1): string {
  // green → amber → red as risk rises
  const t = Math.min(1, score / 100);
  const r = Math.round(88 + t * (255 - 88));
  const g = Math.round(201 - t * (201 - 93));
  const b = Math.round(138 - t * (138 - 93));
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

export function SceneMap({ points }: { points: HeatPoint[] }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(520);

  // track the panel's real width so the canvas never gets CSS-stretched
  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const observer = new ResizeObserver((entries) => {
      const w = Math.round(entries[0].contentRect.width);
      if (w > 0) setWidth(w);
    });
    observer.observe(wrap);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // render at the device's native pixel density (crisp on Retina)
    const dpr = window.devicePixelRatio || 1;
    const W = width;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    if (points.length === 0) return;

    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const minX = Math.min(...xs) - 1;
    const maxX = Math.max(...xs) + 1;
    const minY = Math.min(...ys) - 1;
    const maxY = Math.max(...ys) + 1;
    const sx = (x: number) => PAD + ((x - minX) / (maxX - minX)) * (W - 2 * PAD);
    const sy = (y: number) => PAD + ((y - minY) / (maxY - minY)) * (H - 2 * PAD);

    // meter grid
    ctx.strokeStyle = "#2a3240";
    ctx.fillStyle = "#8b95a7";
    ctx.font = "11px -apple-system, 'Segoe UI', sans-serif";
    ctx.lineWidth = 1;
    const stepX = Math.max(1, Math.round((maxX - minX) / 8));
    for (let m = Math.ceil(minX); m <= maxX; m += stepX) {
      ctx.beginPath();
      ctx.moveTo(sx(m), PAD / 2);
      ctx.lineTo(sx(m), H - PAD / 2);
      ctx.stroke();
      ctx.fillText(`${m}m`, sx(m) - 8, H - 8);
    }
    const stepY = Math.max(1, Math.round((maxY - minY) / 6));
    for (let m = Math.ceil(minY); m <= maxY; m += stepY) {
      ctx.beginPath();
      ctx.moveTo(PAD / 2, sy(m));
      ctx.lineTo(W - PAD / 2, sy(m));
      ctx.stroke();
      ctx.fillText(`${m}m`, 6, sy(m) + 3);
    }

    // heat blobs: translucent halos stack up where events cluster
    for (const p of points) {
      const cx = sx(p.x);
      const cy = sy(p.y);
      const radius = 14 + (p.score / 100) * 14;
      const grad = ctx.createRadialGradient(cx, cy, 2, cx, cy, radius);
      grad.addColorStop(0, scoreColor(p.score, 0.55));
      grad.addColorStop(1, scoreColor(p.score, 0));
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fill();
    }
    // crisp centers on top
    for (const p of points) {
      ctx.fillStyle = scoreColor(p.score);
      ctx.beginPath();
      ctx.arc(sx(p.x), sy(p.y), 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [points, width]);

  return (
    <div className="scene-map" ref={wrapRef}>
      <canvas ref={ref} style={{ width: "100%", height: H }} />
      <div className="hint">
        Each halo is one near miss at its worst moment, plotted on the
        calibrated ground plane. Brighter red = higher risk; overlapping
        halos reveal hotspots.
      </div>
    </div>
  );
}
