"""Road user detection and tracking built on YOLO11 + ByteTrack.

This is the perception core of Reflex: given video frames, it finds every
road user and keeps a stable identity on each one across frames.
"""

from dataclasses import dataclass, field

import numpy as np
import torch
from ultralytics import YOLO

# COCO class ids for the road users Reflex cares about
ROAD_USER_CLASSES = {
    0: "pedestrian",   # COCO "person"
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Vulnerable road users get heavier weighting later in the risk engine
VULNERABLE = {"pedestrian", "bicycle", "motorcycle"}


@dataclass
class TrackedUser:
    """One road user in one frame, with a persistent track id."""

    track_id: int
    class_name: str
    confidence: float
    box_xyxy: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.box_xyxy
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def bottom_center(self) -> tuple[float, float]:
        """Ground contact point — where the user touches the road.

        Used for homography projection later: the bottom of the box sits on
        the ground plane, the center does not.
        """
        x1, _, x2, y2 = self.box_xyxy
        return ((x1 + x2) / 2, y2)


@dataclass
class FrameResult:
    frame_index: int
    users: list[TrackedUser] = field(default_factory=list)


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class RoadUserTracker:
    """Wraps YOLO11 + ByteTrack into a simple per-frame API."""

    def __init__(self, weights: str = "yolo11n.pt", confidence: float = 0.3):
        self.model = YOLO(weights)
        self.confidence = confidence
        self.device = pick_device()

    def track_frame(self, frame: np.ndarray, frame_index: int) -> FrameResult:
        results = self.model.track(
            frame,
            persist=True,
            classes=list(ROAD_USER_CLASSES),
            conf=self.confidence,
            # one physical object = one box: without this, the same vehicle
            # often gets both a "car" and a "truck" box, spawning a phantom
            # second road user
            agnostic_nms=True,
            tracker="bytetrack.yaml",
            device=self.device,
            verbose=False,
        )
        result = FrameResult(frame_index=frame_index)
        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return result
        for box, track_id, cls, conf in zip(
            boxes.xyxy.cpu().numpy(),
            boxes.id.cpu().numpy().astype(int),
            boxes.cls.cpu().numpy().astype(int),
            boxes.conf.cpu().numpy(),
        ):
            result.users.append(
                TrackedUser(
                    track_id=int(track_id),
                    class_name=ROAD_USER_CLASSES[int(cls)],
                    confidence=float(conf),
                    box_xyxy=tuple(box),
                )
            )
        return result
