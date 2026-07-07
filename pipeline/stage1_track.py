"""Stage 1 pipeline: video in → annotated video out.

Runs detection + tracking over a video, draws boxes, ids, and motion
trails on every frame, and writes the result next to a summary of what
it saw. Usage:

    python -m pipeline.stage1_track data/raw/traffic.mp4
"""

import argparse
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.detection.detector import RoadUserTracker, VULNERABLE

# BGR colors per class — warm colors for vulnerable users so they pop
CLASS_COLORS = {
    "pedestrian": (0, 64, 255),
    "bicycle": (0, 160, 255),
    "motorcycle": (0, 220, 255),
    "car": (80, 200, 120),
    "bus": (200, 160, 60),
    "truck": (220, 120, 160),
}
TRAIL_LENGTH = 30  # frames of motion history drawn behind each user


def annotate(frame, result, trails):
    for user in result.users:
        x1, y1, x2, y2 = (int(v) for v in user.box_xyxy)
        color = CLASS_COLORS[user.class_name]
        thickness = 2 if user.class_name in VULNERABLE else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        label = f"#{user.track_id} {user.class_name}"
        cv2.putText(frame, label, (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        trail = trails[user.track_id]
        trail.append(tuple(int(v) for v in user.bottom_center))
        for a, b in zip(trail, list(trail)[1:]):
            cv2.line(frame, a, b, color, 1, cv2.LINE_AA)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Reflex stage 1: detect + track")
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    out_path = args.out or Path("data/output") / f"{args.video.stem}_annotated.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        sys.exit(f"could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    tracker = RoadUserTracker(weights=args.weights)
    print(f"device: {tracker.device} | input: {args.video} ({width}x{height} @ {fps:.0f}fps)")

    trails = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))
    seen_ids: dict[str, set[int]] = defaultdict(set)
    frame_index = 0
    started = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and frame_index >= args.max_frames):
            break
        result = tracker.track_frame(frame, frame_index)
        for user in result.users:
            seen_ids[user.class_name].add(user.track_id)
        writer.write(annotate(frame, result, trails))
        frame_index += 1

    elapsed = time.perf_counter() - started
    cap.release()
    writer.release()

    print(f"\nprocessed {frame_index} frames in {elapsed:.1f}s "
          f"({frame_index / elapsed:.1f} fps on {tracker.device})")
    print(f"annotated video: {out_path}")
    print("\nunique road users tracked:")
    for name, ids in sorted(seen_ids.items()):
        print(f"  {name:<12} {len(ids)}")
    print(f"  {'total':<12} {sum(len(v) for v in seen_ids.values())}")


if __name__ == "__main__":
    main()
