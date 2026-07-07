"""Stage 2 pipeline: video in → real world motion out.

Adds calibration and physics on top of stage 1: every road user's ground
position is projected to meters, smoothed with a Kalman filter, and
annotated with live speed. Also renders a top-down map of all
trajectories in real world coordinates. Usage:

    python -m pipeline.stage2_physics data/raw/mixed_road_users.mp4 \
        --calibration calibrations/mixed_road_users.json
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.calibration.homography import GroundPlane
from ml.detection.detector import RoadUserTracker
from ml.tracking.trajectory import TrajectoryBuilder
from pipeline.stage1_track import CLASS_COLORS


def draw(frame, result, builder):
    for user in result.users:
        x1, y1, x2, y2 = (int(v) for v in user.box_xyxy)
        color = CLASS_COLORS[user.class_name]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        traj = builder.trajectories[user.track_id]
        kmh = traj.speed_mps * 3.6
        label = f"#{user.track_id} {user.class_name} {kmh:.0f} km/h"
        cv2.putText(frame, label, (x1, max(14, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return frame


def plot_trajectories(builder, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    for traj in builder.trajectories.values():
        if traj.duration < 1.0:
            continue  # skip flickers
        xs, ys = zip(*traj.positions)
        rgb = tuple(c / 255 for c in reversed(CLASS_COLORS[traj.class_name]))
        ax.plot(xs, ys, color=rgb, linewidth=1.5)
        ax.annotate(f"#{traj.track_id} {traj.class_name}", (xs[-1], ys[-1]),
                    fontsize=7, color=rgb)
    ax.set_xlabel("meters")
    ax.set_ylabel("meters")
    ax.set_title("Top-down trajectories (ground plane)")
    ax.set_aspect("equal")
    ax.invert_yaxis()  # world Y grows toward camera; draw it downward
    ax.grid(alpha=0.3)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reflex stage 2: motion physics")
    parser.add_argument("video", type=Path)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--weights", default="yolo11n.pt")
    args = parser.parse_args()

    plane = GroundPlane.from_file(args.calibration)
    print(f"calibration reprojection error: {plane.reprojection_error():.3f} m")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video = Path("data/output") / f"{args.video.stem}_physics.mp4"
    out_plot = Path("data/output") / f"{args.video.stem}_trajectories.png"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    tracker = RoadUserTracker(weights=args.weights)
    builder = TrajectoryBuilder(plane)

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result = tracker.track_frame(frame, frame_index)
        builder.update(result, t=frame_index / fps)
        writer.write(draw(frame, result, builder))
        frame_index += 1
    cap.release()
    writer.release()

    plot_trajectories(builder, out_plot)

    print(f"annotated video: {out_video}")
    print(f"trajectory map:  {out_plot}\n")
    print(f"{'id':>4} {'class':<12} {'secs':>6} {'avg km/h':>9} {'max km/h':>9}")
    for traj in builder.trajectories.values():
        if traj.duration < 1.0:
            continue
        s = traj.stats()
        print(f"{s['track_id']:>4} {s['class']:<12} {s['duration_s']:>6} "
              f"{s['avg_speed_kmh']:>9} {s['max_speed_kmh']:>9}")


if __name__ == "__main__":
    main()
