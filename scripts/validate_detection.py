"""Measure detection accuracy against Urban Tracker ground truth.

The Sherbrooke annotations (Polytrack format) hand-label every road user
inside a scene mask for frames 2754-3754. This script runs the Reflex
detector over the same frames and scores it: a detection matching a
ground truth box at IoU >= 0.5 is a true positive.

Fairness rules (matching how the dataset was built):
- The GT annotates MOVING road users only — a background-subtraction
  tracker never sees parked cars. So detections are scored only if their
  track actually moved (>3 m on the ground plane); stationary detections
  (parked cars) are ignored, not false positives.
- Detections whose ground point falls outside the annotation mask are
  ignored (the humans didn't label there, so we can't call them wrong).
- Ground truth objects typed "unknown" are ignore regions: matching them
  neither rewards nor penalizes.
- Class-aware matching: vehicle detections (car/truck/bus) match GT cars,
  pedestrian detections match GT pedestrians.
- Primary matching follows the dataset's own protocol: box centroids
  within 90 px (ReadMe: "Metrics tool must be use with ABSDIST 90").
  A stricter IoU >= 0.5 score is reported alongside.

Usage:
    python scripts/validate_detection.py data/raw/sherbrooke.avi \
        --gt data/raw/sherbrooke_annotations/sherbrooke_annotations/sherbrooke_gt.sqlite \
        --mask data/raw/sherbrooke_mask.png \
        --calibration calibrations/sherbrooke.json
"""

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.calibration.homography import GroundPlane
from ml.detection.detector import RoadUserTracker

GROUPS = {"car": "vehicle", "truck": "vehicle", "bus": "vehicle",
          "motorcycle": "vehicle", "bicycle": "vehicle",
          "pedestrian": "pedestrian"}
IOU_THRESHOLD = 0.5
ABSDIST = 90.0           # px, the dataset's own matching protocol
MIN_TRACK_TRAVEL_M = 3.0  # ground-plane displacement for a track to count as moving
MIN_TRACK_TRAVEL_PX = 30.0  # ...and in pixels, so calibration extrapolation
                            # can't promote a jittering parked car to "moving"


def load_ground_truth(gt_path: Path):
    """Returns ({frame: [(group, box), ...]}, {object_id: (group, {frame: box})}),
    with 'ignore' for unknown-typed objects."""
    conn = sqlite3.connect(gt_path)
    rows = conn.execute(
        """select b.frame_number, t.type_string, o.object_id,
                  b.x_top_left, b.y_top_left, b.x_bottom_right, b.y_bottom_right
           from bounding_boxes b
           join objects o using(object_id)
           join objects_type t using(road_user_type)""").fetchall()
    conn.close()
    gt = defaultdict(list)
    objects: dict = {}
    for frame, type_string, obj_id, x1, y1, x2, y2 in rows:
        group = {"car": "vehicle", "pedestrians": "pedestrian"}.get(
            type_string, "ignore")
        gt[frame].append((group, (x1, y1, x2, y2)))
        objects.setdefault(obj_id, (group, {}))[1][frame] = (x1, y1, x2, y2)
    return gt, objects


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)
    return inter / union


def centroid(box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def score(per_frame_detections, gt, match_fn) -> dict:
    """Greedy per-frame matching; returns tp/fp/fn per group."""
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for frame_number, truths in gt.items():
        detections = per_frame_detections.get(frame_number, [])
        matched_gt = set()
        for d_group, d_box in detections:
            best_i, best_v = -1, 0.0
            for i, (t_group, t_box) in enumerate(truths):
                if i in matched_gt:
                    continue
                if t_group != "ignore" and t_group != d_group:
                    continue
                v = match_fn(d_box, t_box)
                if v > best_v:
                    best_i, best_v = i, v
            if best_i >= 0:
                matched_gt.add(best_i)
                if truths[best_i][0] != "ignore":
                    tp[d_group] += 1
            else:
                fp[d_group] += 1
        for i, (t_group, _) in enumerate(truths):
            if t_group != "ignore" and i not in matched_gt:
                fn[t_group] += 1
    return {"tp": tp, "fp": fp, "fn": fn}


def report(name: str, counts: dict) -> None:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    print(f"\n[{name}]")
    print(f"{'group':<12} {'TP':>7} {'FP':>6} {'FN':>6} "
          f"{'precision':>10} {'recall':>8} {'F1':>7}")
    totals = [0, 0, 0]
    for group in ("vehicle", "pedestrian", "overall"):
        if group == "overall":
            t, f_, n = totals
        else:
            t, f_, n = tp[group], fp[group], fn[group]
            totals = [totals[0] + t, totals[1] + f_, totals[2] + n]
        p = t / (t + f_) if t + f_ else 0
        r = t / (t + n) if t + n else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        print(f"{group:<12} {t:>7} {f_:>6} {n:>6} {p:>10.3f} {r:>8.3f} {f1:>7.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="detection accuracy vs ground truth")
    parser.add_argument("video", type=Path)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--weights", default="yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=0.3)
    args = parser.parse_args()

    gt, gt_objects = load_ground_truth(args.gt)
    frames = sorted(gt)
    mask = cv2.imread(str(args.mask), cv2.IMREAD_GRAYSCALE)
    plane = GroundPlane.from_file(args.calibration)

    tracker = RoadUserTracker(weights=args.weights, confidence=args.confidence)
    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frames[0])

    # Pass 1: collect every detection with its track id, and each track's
    # ground-plane path, so we can tell moving road users from parked ones.
    raw = defaultdict(list)          # frame -> [(track_id, group, box)]
    track_path = defaultdict(list)   # track_id -> [ground xy]
    for frame_number in range(frames[0], frames[-1] + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if frame_number not in gt:
            continue
        result = tracker.track_frame(frame, frame_number)
        for user in result.users:
            gx, gy = user.bottom_center
            gx = int(np.clip(gx, 0, mask.shape[1] - 1))
            gy = int(np.clip(gy, 0, mask.shape[0] - 1))
            if mask[gy, gx] == 0:
                continue  # outside the annotated region
            raw[frame_number].append(
                (user.track_id, GROUPS[user.class_name], user.box_xyxy))
            track_path[user.track_id].append(
                (plane.to_world([user.bottom_center])[0], user.bottom_center))
    cap.release()

    # Moving = displacement above threshold in BOTH ground meters and raw
    # pixels (medians of the first/last few observations, so box jitter
    # doesn't fake motion). The GT never labels parked vehicles, so
    # stationary tracks are excluded from scoring rather than punished.
    moving = set()
    for tid, path in track_path.items():
        if len(path) < 3:
            continue
        k = min(5, len(path) // 2)
        world = np.array([p[0] for p in path])
        pixel = np.array([p[1] for p in path])
        world_d = float(np.linalg.norm(
            np.median(world[-k:], axis=0) - np.median(world[:k], axis=0)))
        pixel_d = float(np.linalg.norm(
            np.median(pixel[-k:], axis=0) - np.median(pixel[:k], axis=0)))
        if world_d >= MIN_TRACK_TRAVEL_M and pixel_d >= MIN_TRACK_TRAVEL_PX:
            moving.add(tid)

    # Parked-vehicle zones: a spot occupied by a vehicle detection for most
    # of the clip is a parking space, not traffic — the motion-based GT
    # cannot see it, so neither should the scorer. (Track-id churn along a
    # parked row defeats the per-track filter; occupancy is id-agnostic.)
    n_frames = len(raw)
    cell = 16
    occupancy = defaultdict(int)
    for dets in raw.values():
        seen = set()
        for _, group, box in dets:
            if group != "vehicle":
                continue
            cx, cy = centroid(box)
            seen.add((int(cx // cell), int(cy // cell)))
        for c in seen:
            occupancy[c] += 1
    parked_cells = {c for c, n in occupancy.items() if n > 0.5 * n_frames}

    def is_parked(box) -> bool:
        cx, cy = centroid(box)
        return (int(cx // cell), int(cy // cell)) in parked_cells

    per_frame = {
        f: [(g, box) for tid, g, box in dets
            if tid in moving and not (g == "vehicle" and is_parked(box))]
        for f, dets in raw.items()
    }

    print(f"frames evaluated: {len(frames)} | conf >= {args.confidence} | "
          f"tracks: {len(track_path)} total, {len(moving)} moving "
          f"(>= {MIN_TRACK_TRAVEL_M}m travel)")

    report(f"dataset protocol: centroid distance <= {ABSDIST:.0f}px",
           score(per_frame, gt,
                 lambda d, t: max(0.0, ABSDIST - float(np.hypot(
                     *(np.array(centroid(d)) - np.array(centroid(t)))))) / ABSDIST
                 if np.hypot(*(np.array(centroid(d)) - np.array(centroid(t)))) <= ABSDIST
                 else 0.0))
    report(f"strict: IoU >= {IOU_THRESHOLD}",
           score(per_frame, gt,
                 lambda d, t: iou(d, t) if iou(d, t) >= IOU_THRESHOLD else 0.0))

    # Track-level coverage: for each annotated road user, in what fraction
    # of their frames did we have a matching detection? "Followed" means
    # covered for more than half their time on screen.
    print("\n[track-level coverage: annotated road users we followed]")
    followed = defaultdict(int)
    total = defaultdict(int)
    for obj_id, (group, boxes) in sorted(gt_objects.items()):
        if group == "ignore":
            continue
        hits = 0
        for frame, t_box in boxes.items():
            tc = np.array(centroid(t_box))
            if any(g == group and
                   float(np.hypot(*(np.array(centroid(b)) - tc))) <= ABSDIST
                   for g, b in per_frame.get(frame, [])):
                hits += 1
        coverage = hits / len(boxes)
        total[group] += 1
        followed[group] += coverage > 0.5
        print(f"  object {obj_id:>3} ({group:<10}) coverage {coverage:5.1%} "
              f"over {len(boxes)} frames")
    for group in ("vehicle", "pedestrian"):
        if total[group]:
            print(f"{group}: followed {followed[group]}/{total[group]}")
    n_f = sum(followed.values())
    n_t = sum(total.values())
    print(f"overall: followed {n_f}/{n_t} = {n_f / n_t:.1%}")


if __name__ == "__main__":
    main()
