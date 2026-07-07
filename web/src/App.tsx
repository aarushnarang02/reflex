import { useEffect, useMemo, useState } from "react";
import {
  api,
  type HeatPoint,
  type NearMissEvent,
  type Stats,
  type VideoSummary,
} from "./api";
import { EventDetail } from "./components/EventDetail";
import { EventsTable } from "./components/EventsTable";
import { SceneMap } from "./components/SceneMap";
import { Trends } from "./components/Trends";

export default function App() {
  const [videos, setVideos] = useState<VideoSummary[]>([]);
  const [videoId, setVideoId] = useState<number | null>(null);
  const [events, setEvents] = useState<NearMissEvent[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [heat, setHeat] = useState<HeatPoint[]>([]);
  const [selected, setSelected] = useState<NearMissEvent | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .videos()
      .then((v) => {
        setVideos(v);
        const requested = Number(new URLSearchParams(window.location.search).get("video"));
        const initial = v.find((x) => x.id === requested) ?? v[0];
        if (initial) setVideoId(initial.id);
      })
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (videoId == null) return;
    setSelected(null);
    Promise.all([api.events(videoId), api.stats(videoId), api.heatmap(videoId)])
      .then(([ev, st, hm]) => {
        setEvents(ev);
        setStats(st);
        setHeat(hm);
        const requested = Number(new URLSearchParams(window.location.search).get("event"));
        setSelected(ev.find((e) => e.id === requested) ?? ev[0] ?? null);
      })
      .catch((e) => setError(String(e)));
  }, [videoId]);

  const video = useMemo(
    () => videos.find((v) => v.id === videoId) ?? null,
    [videos, videoId],
  );

  return (
    <div className="app">
      <div className="header">
        <h1>REFLEX</h1>
        <span className="tag">collision risk dashboard</span>
        {error && <span style={{ color: "var(--danger)" }}>{error}</span>}
      </div>

      <div className="video-picker">
        <label htmlFor="scene-select">Scene</label>
        <select
          id="scene-select"
          value={videoId ?? ""}
          onChange={(e) => setVideoId(Number(e.target.value))}
        >
          {videos.map((v) => (
            <option key={v.id} value={v.id}>
              {v.name} — {v.n_events} events
            </option>
          ))}
        </select>
      </div>

      <div className="cards">
        <div className="card">
          <div className="label">road users tracked</div>
          <div className="value">{video?.n_tracks ?? "—"}</div>
        </div>
        <div className="card">
          <div className="label">near miss events</div>
          <div className="value">{stats?.total_events ?? "—"}</div>
        </div>
        <div className="card">
          <div className="label">worst risk score</div>
          <div className="value">{stats ? stats.max_score.toFixed(0) : "—"}</div>
        </div>
        <div className="card">
          <div className="label">footage analyzed</div>
          <div className="value">
            {video ? `${(video.duration_s / 60).toFixed(1)} min` : "—"}
          </div>
        </div>
      </div>

      <div className="grid-2">
        <div className="panel">
          <h2>Event replay</h2>
          <EventDetail event={selected} />
        </div>
        <div className="panel">
          <h2>Close call hotspots</h2>
          <SceneMap points={heat} />
        </div>
      </div>

      <div className="grid-2">
        <div className="panel">
          <h2>Flagged events</h2>
          <div style={{ maxHeight: 420, overflowY: "auto" }}>
            <EventsTable
              events={events}
              selectedId={selected?.id ?? null}
              onSelect={setSelected}
            />
          </div>
        </div>
        <div className="panel">
          <h2>Trends</h2>
          <Trends stats={stats} />
        </div>
      </div>
    </div>
  );
}
