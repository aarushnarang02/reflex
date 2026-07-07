"""Tile event peak moments into contact sheets for fast human review.

Precision review means a person looks at every flagged event and rules
genuine / false. Opening clips one by one is slow; this script grabs each
event's worst frame from the risk-annotated video and tiles them into
numbered grids, so a reviewer can triage dozens at a glance and only
open clips for the ambiguous ones.

    python scripts/event_montage.py --video-id 3
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api import models

COLS, ROWS = 4, 3          # 12 events per sheet
TILE_W, TILE_H = 400, 300
LABEL_H = 26


def main() -> None:
    parser = argparse.ArgumentParser(description="build event review sheets")
    parser.add_argument("--video-id", type=int, required=True)
    parser.add_argument("--db", default=None)
    args = parser.parse_args()

    engine = models.get_engine(args.db)
    with models.session(engine) as s:
        video = s.get(models.Video, args.video_id)
        events = s.query(models.Event).filter_by(video_id=args.video_id) \
            .order_by(models.Event.risk_score.desc()).all()
    if not events:
        sys.exit("no events for this video")

    stem = Path(video.path).stem
    annotated = Path("data/output") / f"{stem}_risk.mp4"
    cap = cv2.VideoCapture(str(annotated))
    out_dir = Path("data/output/review") / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    per_sheet = COLS * ROWS
    for sheet_i in range(0, len(events), per_sheet):
        batch = events[sheet_i:sheet_i + per_sheet]
        sheet = np.zeros((ROWS * (TILE_H + LABEL_H), COLS * TILE_W, 3), np.uint8)
        for k, ev in enumerate(batch):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(ev.t_peak * video.fps))
            ok, frame = cap.read()
            if not ok:
                continue
            tile = cv2.resize(frame, (TILE_W, TILE_H))
            r, c = divmod(k, COLS)
            y0 = r * (TILE_H + LABEL_H)
            x0 = c * TILE_W
            sheet[y0:y0 + TILE_H, x0:x0 + TILE_W] = tile
            label = (f"ev{ev.id} {ev.class_a[:3]}x{ev.class_b[:3]} "
                     f"s{ev.risk_score:.0f} ttc{ev.min_ttc_s or 9:.1f} "
                     f"d{ev.min_distance_m:.1f}m c{ev.max_closing_kmh:.0f}")
            cv2.putText(sheet, label, (x0 + 6, y0 + TILE_H + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 220, 255), 1,
                        cv2.LINE_AA)
        path = out_dir / f"sheet_{sheet_i // per_sheet:02d}.png"
        cv2.imwrite(str(path), sheet)
        print(f"wrote {path} ({len(batch)} events)")
    cap.release()


if __name__ == "__main__":
    main()
