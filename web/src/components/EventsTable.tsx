// Filterable list of flagged near misses, worst first.

import type { NearMissEvent } from "../api";

function fmtTime(s: number): string {
  const m = Math.floor(s / 60);
  return `${m}:${(s % 60).toFixed(0).padStart(2, "0")}`;
}

function scoreClass(score: number): string {
  if (score >= 60) return "score high";
  if (score >= 40) return "score mid";
  return "score low";
}

export function EventsTable({
  events,
  selectedId,
  onSelect,
}: {
  events: NearMissEvent[];
  selectedId: number | null;
  onSelect: (e: NearMissEvent) => void;
}) {
  if (events.length === 0)
    return <div className="empty">No events for this selection.</div>;
  return (
    <table>
      <thead>
        <tr>
          <th>at</th>
          <th>who</th>
          <th>risk</th>
          <th>TTC</th>
          <th>PET</th>
          <th>min dist</th>
          <th>closing</th>
        </tr>
      </thead>
      <tbody>
        {events.map((e) => (
          <tr
            key={e.id}
            className={e.id === selectedId ? "selected" : ""}
            onClick={() => onSelect(e)}
          >
            <td>{fmtTime(e.t_peak)}</td>
            <td>
              {e.class_a} × {e.class_b}
            </td>
            <td>
              <span className={scoreClass(e.risk_score)}>
                {e.risk_score.toFixed(0)}
              </span>
            </td>
            <td>{e.min_ttc_s != null ? `${e.min_ttc_s}s` : "—"}</td>
            <td>{e.pet_s != null ? `${e.pet_s}s` : "—"}</td>
            <td>{e.min_distance_m}m</td>
            <td>{e.max_closing_kmh} km/h</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
