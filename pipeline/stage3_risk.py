"""Stage 3 pipeline: video in → near miss events in PostgreSQL.

The full Reflex loop: perceive, ground, score, persist. Produces an
annotated video with risk overlays, evidence clips per event, and rows
in the events table. Usage:

    python -m pipeline.stage3_risk data/raw/mixed_road_users.mp4 \
        --calibration calibrations/mixed_road_users.json
"""

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api import models
from ml.calibration.homography import GroundPlane
from ml.detection.detector import RoadUserTracker
from ml.risk.engine import RiskEngine
from ml.tracking.trajectory import TrajectoryBuilder
from pipeline import clipper
from pipeline.stage2_physics import draw as draw_physics


def draw_risk(frame, result, overlays):
    boxes = {u.track_id: u for u in result.users}
    for ov in overlays:
        a, b = boxes.get(ov["a"]), boxes.get(ov["b"])
        if a is None or b is None:
            continue
        ca = tuple(int(v) for v in a.center)
        cb = tuple(int(v) for v in b.center)
        danger = min(1.0, ov["score"] / 100.0)
        color = (0, int(200 * (1 - danger)), 255)  # yellow → red as risk rises
        cv2.line(frame, ca, cb, color, 2, cv2.LINE_AA)
        mid = ((ca[0] + cb[0]) // 2, (ca[1] + cb[1]) // 2)
        ttc = f" TTC {ov['ttc']:.1f}s" if math.isfinite(ov["ttc"]) else ""
        cv2.putText(frame, f"risk {ov['score']:.0f}{ttc}", mid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Reflex stage 3: risk engine")
    parser.add_argument("video", type=Path)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--name", default=None, help="display name shown on the dashboard")
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--threshold", type=float, default=30.0)
    parser.add_argument("--include-ped-ped", action="store_true",
                        help="also score pedestrian-pedestrian pairs (crowd analysis)")
    parser.add_argument("--db", default=None, help="SQLAlchemy URL (default: local postgres)")
    args = parser.parse_args()

    plane = GroundPlane.from_file(args.calibration)
    engine_db = models.get_engine(args.db)
    models.init_db(engine_db)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video = Path("data/output") / f"{args.video.stem}_risk.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    tracker = RoadUserTracker(weights=args.weights)
    builder = TrajectoryBuilder(plane)
    risk = RiskEngine(threshold=args.threshold,
                      score_pedestrian_pairs=args.include_ped_ped,
                      in_valid_region=plane.is_valid)

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_index / fps
        result = tracker.track_frame(frame, frame_index)
        builder.update(result, t)
        active = {u.track_id: builder.trajectories[u.track_id] for u in result.users}
        overlays = risk.update(active, t)
        frame = draw_physics(frame, result, builder)
        frame = draw_risk(frame, result, overlays)
        writer.write(frame)
        frame_index += 1
    cap.release()
    writer.release()
    risk.finish(builder.trajectories)

    # ---- persist everything ----
    with models.session(engine_db) as db:
        video_row = models.Video(
            path=str(args.video), name=args.name, fps=fps, width=width,
            height=height, frame_count=frame_index,
            calibration=str(args.calibration))
        db.add(video_row)
        db.flush()

        track_rows = {}
        for traj in builder.trajectories.values():
            s = traj.stats()
            row = models.Track(
                video_id=video_row.id, track_num=traj.track_id,
                class_name=traj.class_name, t_first=traj.times[0],
                t_last=traj.times[-1], avg_speed_kmh=s["avg_speed_kmh"],
                max_speed_kmh=s["max_speed_kmh"], n_observations=len(traj.times))
            db.add(row)
            track_rows[traj.track_id] = row
        db.flush()

        clip_dir = Path("data/clips") / args.video.stem
        for i, ev in enumerate(risk.events):
            clip = clipper.extract_clip(
                out_video, ev.t_start, ev.t_end,
                clip_dir / f"event_{i:03d}_{ev.class_a}_{ev.class_b}.mp4")
            db.add(models.Event(
                video_id=video_row.id,
                track_a_id=track_rows[ev.track_a].id,
                track_b_id=track_rows[ev.track_b].id,
                class_a=ev.class_a, class_b=ev.class_b,
                t_start=ev.t_start, t_peak=ev.t_peak, t_end=ev.t_end,
                risk_score=ev.peak_score,
                min_ttc_s=None if math.isinf(ev.min_ttc) else ev.min_ttc,
                pet_s=None if math.isinf(ev.pet) else ev.pet,
                min_distance_m=ev.min_distance,
                max_closing_kmh=ev.max_closing_kmh,
                peak_x_m=ev.peak_location[0], peak_y_m=ev.peak_location[1],
                clip_path=str(clip)))
        db.commit()
        video_id = video_row.id

    # ---- report ----
    print(f"annotated video: {out_video}")
    print(f"video id {video_id}: {len(builder.trajectories)} tracks, "
          f"{len(risk.events)} near miss events\n")
    if risk.events:
        print(f"{'pair':<24} {'peak':>5} {'TTC s':>6} {'PET s':>6} {'min m':>6} {'close km/h':>11}")
        for ev in sorted(risk.events, key=lambda e: -e.peak_score):
            ttc = f"{ev.min_ttc:.2f}" if math.isfinite(ev.min_ttc) else "-"
            pet = f"{ev.pet:.2f}" if math.isfinite(ev.pet) else "-"
            print(f"#{ev.track_a} {ev.class_a} x #{ev.track_b} {ev.class_b:<10} "
                  f"{ev.peak_score:>5.0f} {ttc:>6} {pet:>6} "
                  f"{ev.min_distance:>6.1f} {ev.max_closing_kmh:>11.1f}")
    else:
        # sparse footage: show the closest interactions anyway
        log = sorted(risk.interaction_log, key=lambda r: -r["score"])[:5]
        print("no events above threshold; closest interactions:")
        for r in log:
            ttc = f"{r['ttc']:.2f}s" if math.isfinite(r["ttc"]) else "-"
            print(f"  t={r['t']:.1f}s  #{r['a']} x #{r['b']}  score {r['score']:.0f}  "
                  f"dist {r['distance']:.1f}m  TTC {ttc}  closing {r['closing_kmh']:.0f} km/h")


if __name__ == "__main__":
    main()
