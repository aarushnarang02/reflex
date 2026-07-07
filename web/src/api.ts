// Typed client for the Reflex API.

export interface VideoSummary {
  id: number;
  path: string;
  name: string;
  fps: number;
  duration_s: number;
  width: number;
  height: number;
  processed_at: string;
  n_tracks: number;
  n_events: number;
}

export interface NearMissEvent {
  id: number;
  video_id: number;
  class_a: string;
  class_b: string;
  t_start: number;
  t_peak: number;
  t_end: number;
  risk_score: number;
  min_ttc_s: number | null;
  pet_s: number | null;
  min_distance_m: number;
  max_closing_kmh: number;
  peak_x_m: number;
  peak_y_m: number;
  has_clip: boolean;
}

export interface Stats {
  total_events: number;
  max_score: number;
  mean_score: number;
  by_pair: { pair: string; count: number }[];
  by_time: { t: number; count: number }[];
}

export interface HeatPoint {
  x: number;
  y: number;
  score: number;
  id: number;
}

async function get<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} → ${res.status}`);
  return res.json();
}

export const api = {
  videos: () => get<VideoSummary[]>("/api/videos"),
  events: (videoId?: number, minScore = 0) =>
    get<NearMissEvent[]>(
      `/api/events?min_score=${minScore}` +
        (videoId != null ? `&video_id=${videoId}` : ""),
    ),
  stats: (videoId?: number) =>
    get<Stats>(`/api/stats${videoId != null ? `?video_id=${videoId}` : ""}`),
  heatmap: (videoId?: number) =>
    get<HeatPoint[]>(
      `/api/heatmap${videoId != null ? `?video_id=${videoId}` : ""}`,
    ),
  clipUrl: (eventId: number) => `/api/events/${eventId}/clip`,
};
