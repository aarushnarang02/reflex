"""End-to-end pipeline throughput benchmark.

Measures honest frames per second over the full loop — video decode,
detection, tracking, ground-plane projection, Kalman update, and risk
scoring — with no video writing (that's an output cost, not a pipeline
cost). Reports a warmup-excluded average over the whole run.

Local (Apple Silicon / CPU):
    python scripts/benchmark.py data/raw/sherbrooke.avi \
        --calibration calibrations/sherbrooke.json

TensorRT (NVIDIA GPU, e.g. a RunPod/Lambda/Colab session):
    python scripts/benchmark.py ... --engine        # exports once, then benchmarks
The TensorRT engine build requires CUDA; on non-NVIDIA machines the
flag explains itself and exits.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.calibration.homography import GroundPlane
from ml.detection.detector import RoadUserTracker
from ml.risk.engine import RiskEngine
from ml.tracking.trajectory import TrajectoryBuilder

WARMUP_FRAMES = 30


def main() -> None:
    parser = argparse.ArgumentParser(description="pipeline throughput benchmark")
    parser.add_argument("video", type=Path)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--engine", action="store_true",
                        help="export to TensorRT and benchmark the engine (NVIDIA only)")
    args = parser.parse_args()

    weights = args.weights
    if args.engine:
        import torch
        if not torch.cuda.is_available():
            sys.exit("TensorRT requires an NVIDIA GPU with CUDA. Run this on a "
                     "cloud GPU (RunPod / Lambda / Colab) — everything else in "
                     "this script is identical there.")
        from ultralytics import YOLO
        engine_path = Path(args.weights).with_suffix(".engine")
        if not engine_path.exists():
            print("exporting TensorRT engine (one-time, takes a few minutes)...")
            YOLO(args.weights).export(format="engine", half=True)
        weights = str(engine_path)

    plane = GroundPlane.from_file(args.calibration)
    tracker = RoadUserTracker(weights=weights)
    builder = TrajectoryBuilder(plane)
    risk = RiskEngine()

    cap = cv2.VideoCapture(str(args.video))
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0

    n = 0
    timed_frames = 0
    started = None
    while n < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        t = n / fps_src
        result = tracker.track_frame(frame, n)
        builder.update(result, t)
        active = {u.track_id: builder.trajectories[u.track_id] for u in result.users}
        risk.update(active, t)
        n += 1
        if n == WARMUP_FRAMES:
            started = time.perf_counter()  # exclude model load + warmup
        elif n > WARMUP_FRAMES:
            timed_frames += 1
    cap.release()
    elapsed = time.perf_counter() - started

    print(f"\nweights:  {weights}  (device: {tracker.device})")
    print(f"frames:   {timed_frames} timed ({WARMUP_FRAMES} warmup excluded)")
    print(f"pipeline: {timed_frames / elapsed:.1f} fps end-to-end "
          f"({1000 * elapsed / timed_frames:.1f} ms/frame)")
    print(f"realtime: {'yes' if timed_frames / elapsed >= fps_src else 'no'} "
          f"(source is {fps_src:.0f} fps)")


if __name__ == "__main__":
    main()
