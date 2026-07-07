// When do close calls happen, and between whom?

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { Stats } from "../api";

const AXIS = { fill: "#8b95a7", fontSize: 11 };

export function Trends({ stats }: { stats: Stats | null }) {
  if (!stats || stats.total_events === 0)
    return <div className="empty">No events to chart.</div>;

  const timeData = stats.by_time.map((b) => ({
    label: `${Math.floor(b.t / 60)}:${(b.t % 60).toString().padStart(2, "0")}`,
    count: b.count,
  }));

  return (
    <div>
      <ResponsiveContainer width="100%" height={150}>
        <BarChart data={timeData}>
          <CartesianGrid stroke="#2a3240" vertical={false} />
          <XAxis dataKey="label" tick={AXIS} axisLine={{ stroke: "#2a3240" }} />
          <YAxis tick={AXIS} axisLine={{ stroke: "#2a3240" }} allowDecimals={false} />
          <Tooltip
            contentStyle={{ background: "#1e2530", border: "1px solid #2a3240" }}
            labelStyle={{ color: "#e6e9ef" }}
          />
          <Bar dataKey="count" fill="#4da3ff" radius={[3, 3, 0, 0]} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
      <div className="hint">Events per 30 seconds of video</div>

      <ResponsiveContainer width="100%" height={Math.max(120, 60 + stats.by_pair.length * 40)}>
        <BarChart data={stats.by_pair} layout="vertical" margin={{ top: 10, right: 20 }}>
          <CartesianGrid stroke="#2a3240" horizontal={false} />
          <XAxis type="number" tick={AXIS} axisLine={{ stroke: "#2a3240" }} allowDecimals={false} />
          <YAxis type="category" dataKey="pair" tick={{ ...AXIS, fontSize: 10 }} width={150} axisLine={{ stroke: "#2a3240" }} />
          <Tooltip
            contentStyle={{ background: "#1e2530", border: "1px solid #2a3240" }}
            labelStyle={{ color: "#e6e9ef" }}
          />
          <Bar dataKey="count" fill="#ffb84d" radius={[0, 3, 3, 0]} isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
      <div className="hint">Events by participant pair</div>
    </div>
  );
}
