"""Database schema: everything Reflex remembers.

Four tables — videos it has watched, road users it has followed, near
miss events it has flagged, and (later) named zones of the scene. Runs
on PostgreSQL; the URL comes from REFLEX_DB_URL so tests can point the
same code at a scratch database.
"""

import os
from datetime import datetime, timezone

from sqlalchemy import ForeignKey, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

DEFAULT_DB_URL = "postgresql+psycopg://localhost:5432/reflex"


class Base(DeclarativeBase):
    pass


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(primary_key=True)
    path: Mapped[str]
    name: Mapped[str | None]  # human-friendly display name for the dashboard
    fps: Mapped[float]
    width: Mapped[int]
    height: Mapped[int]
    frame_count: Mapped[int]
    calibration: Mapped[str | None]
    processed_at: Mapped[datetime] = mapped_column(
        default=lambda: datetime.now(timezone.utc))


class Track(Base):
    __tablename__ = "tracks"

    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"))
    track_num: Mapped[int]          # id assigned by the tracker within this video
    class_name: Mapped[str]
    t_first: Mapped[float]          # seconds from video start
    t_last: Mapped[float]
    avg_speed_kmh: Mapped[float]
    max_speed_kmh: Mapped[float]
    n_observations: Mapped[int]


class Event(Base):
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id"))
    track_a_id: Mapped[int] = mapped_column(ForeignKey("tracks.id"))
    track_b_id: Mapped[int] = mapped_column(ForeignKey("tracks.id"))
    class_a: Mapped[str]
    class_b: Mapped[str]
    t_start: Mapped[float]
    t_peak: Mapped[float]
    t_end: Mapped[float]
    risk_score: Mapped[float]
    min_ttc_s: Mapped[float | None]      # None = no collision course observed
    pet_s: Mapped[float | None]          # None = paths never crossed
    min_distance_m: Mapped[float]
    max_closing_kmh: Mapped[float]
    peak_x_m: Mapped[float]              # ground plane location of worst moment
    peak_y_m: Mapped[float]
    clip_path: Mapped[str | None]


def get_engine(url: str | None = None):
    return create_engine(url or os.environ.get("REFLEX_DB_URL", DEFAULT_DB_URL))


def init_db(engine) -> None:
    Base.metadata.create_all(engine)


def session(engine) -> Session:
    return Session(engine)
