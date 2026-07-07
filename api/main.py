"""Reflex API: serves the dashboard everything it shows.

Read-only view over what the pipeline persisted — videos, tracks, near
miss events — plus the evidence clips themselves. Run with:

    uvicorn api.main:app --reload
"""

import math
import sys
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api import models

PROJECT_ROOT = Path(__file__).resolve().parents[1]

app = FastAPI(title="Reflex", description="Collision risk detection API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

_engine = models.get_engine()


def db() -> Session:
    with models.session(_engine) as s:
        yield s


@app.get("/api/videos")
def list_videos(s: Session = Depends(db)):
    rows = s.execute(
        select(
            models.Video,
            func.count(func.distinct(models.Track.id)).label("tracks"),
            func.count(func.distinct(models.Event.id)).label("events"),
        )
        .outerjoin(models.Track, models.Track.video_id == models.Video.id)
        .outerjoin(models.Event, models.Event.video_id == models.Video.id)
        .group_by(models.Video.id)
        .order_by(models.Video.id)
    ).all()
    return [
        {
            "id": v.id, "path": v.path,
            "name": v.name or Path(v.path).stem.replace("_", " ").title(),
            "fps": v.fps,
            "duration_s": round(v.frame_count / v.fps, 1) if v.fps else 0,
            "width": v.width, "height": v.height,
            "processed_at": v.processed_at.isoformat(),
            "n_tracks": tracks, "n_events": events,
        }
        for v, tracks, events in rows
    ]


@app.get("/api/events")
def list_events(video_id: int | None = None, min_score: float = 0,
                user_class: str | None = None, s: Session = Depends(db)):
    q = select(models.Event).where(models.Event.risk_score >= min_score)
    if video_id is not None:
        q = q.where(models.Event.video_id == video_id)
    if user_class:
        q = q.where((models.Event.class_a == user_class)
                    | (models.Event.class_b == user_class))
    events = s.scalars(q.order_by(models.Event.risk_score.desc())).all()
    return [_event_json(e) for e in events]


@app.get("/api/events/{event_id}")
def get_event(event_id: int, s: Session = Depends(db)):
    ev = s.get(models.Event, event_id)
    if ev is None:
        raise HTTPException(404, "no such event")
    return _event_json(ev)


@app.get("/api/events/{event_id}/clip")
def get_clip(event_id: int, s: Session = Depends(db)):
    ev = s.get(models.Event, event_id)
    if ev is None or not ev.clip_path:
        raise HTTPException(404, "no clip for this event")
    clip = PROJECT_ROOT / ev.clip_path
    if not clip.exists():
        raise HTTPException(404, "clip file missing from disk")
    return FileResponse(clip, media_type="video/mp4")


@app.get("/api/stats")
def stats(video_id: int | None = None, s: Session = Depends(db)):
    q = select(models.Event)
    if video_id is not None:
        q = q.where(models.Event.video_id == video_id)
    events = s.scalars(q).all()

    by_pair: dict[str, int] = {}
    by_time: dict[int, int] = {}
    for e in events:
        pair = " x ".join(sorted([e.class_a, e.class_b]))
        by_pair[pair] = by_pair.get(pair, 0) + 1
        bucket = int(e.t_peak // 30) * 30  # 30-second buckets of video time
        by_time[bucket] = by_time.get(bucket, 0) + 1

    scores = [e.risk_score for e in events]
    return {
        "total_events": len(events),
        "max_score": max(scores, default=0),
        "mean_score": round(sum(scores) / len(scores), 1) if scores else 0,
        "by_pair": [{"pair": k, "count": v}
                    for k, v in sorted(by_pair.items(), key=lambda i: -i[1])],
        "by_time": [{"t": k, "count": v} for k, v in sorted(by_time.items())],
    }


@app.get("/api/heatmap")
def heatmap(video_id: int | None = None, s: Session = Depends(db)):
    """Ground plane locations of every event's worst moment."""
    q = select(models.Event)
    if video_id is not None:
        q = q.where(models.Event.video_id == video_id)
    return [
        {"x": e.peak_x_m, "y": e.peak_y_m, "score": e.risk_score, "id": e.id}
        for e in s.scalars(q).all()
    ]


def _event_json(e: models.Event) -> dict:
    return {
        "id": e.id, "video_id": e.video_id,
        "class_a": e.class_a, "class_b": e.class_b,
        "t_start": round(e.t_start, 2), "t_peak": round(e.t_peak, 2),
        "t_end": round(e.t_end, 2),
        "risk_score": round(e.risk_score, 1),
        "min_ttc_s": None if e.min_ttc_s is None or math.isinf(e.min_ttc_s)
                     else round(e.min_ttc_s, 2),
        "pet_s": None if e.pet_s is None or math.isinf(e.pet_s)
                 else round(e.pet_s, 2),
        "min_distance_m": round(e.min_distance_m, 2),
        "max_closing_kmh": round(e.max_closing_kmh, 1),
        "peak_x_m": round(e.peak_x_m, 2), "peak_y_m": round(e.peak_y_m, 2),
        "has_clip": bool(e.clip_path),
    }
