"""Camera-to-ground-plane calibration.

The camera sees pixels; danger happens in meters. A homography is the 3x3
matrix that maps points on the image's ground plane to real world ground
coordinates. It is built from a handful of point correspondences: places
in the image whose real world geometry is known (lane widths, parking
stall dimensions, crosswalk stripes — road paint is full of standardized
measurements).
"""

import json
from pathlib import Path

import cv2
import numpy as np


class GroundPlane:
    """Maps image pixels to metric ground plane coordinates."""

    def __init__(self, pixel_points: np.ndarray, world_points: np.ndarray):
        if len(pixel_points) < 4:
            raise ValueError("homography needs at least 4 point correspondences")
        self.pixel_points = np.asarray(pixel_points, dtype=np.float64)
        self.world_points = np.asarray(world_points, dtype=np.float64)
        self.matrix, _ = cv2.findHomography(self.pixel_points, self.world_points, 0)
        if self.matrix is None:
            raise ValueError("degenerate point configuration, homography failed")

    @classmethod
    def from_file(cls, path: str | Path) -> "GroundPlane":
        data = json.loads(Path(path).read_text())
        pts = data["points"]
        plane = cls(
            np.array([p["pixel"] for p in pts], dtype=np.float64),
            np.array([p["world"] for p in pts], dtype=np.float64),
        )
        plane.valid_region = data.get("valid_region")
        return plane

    # A homography is only trustworthy near its calibration points; far
    # outside them, small pixel errors become huge metric errors (lanes
    # compress, speeds inflate). The calibration file may declare the
    # world-coordinate box where measurements are reliable.
    valid_region: dict | None = None

    def is_valid(self, x: float, y: float) -> bool:
        if not self.valid_region:
            return True
        x0, x1 = self.valid_region["x"]
        y0, y1 = self.valid_region["y"]
        return x0 <= x <= x1 and y0 <= y <= y1

    def to_world(self, pixels: np.ndarray) -> np.ndarray:
        """Project pixel points (N x 2) onto the ground plane in meters."""
        pts = np.asarray(pixels, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.matrix).reshape(-1, 2)

    def reprojection_error(self) -> float:
        """Mean distance (meters) between calibration points and where the
        fitted homography actually sends them. A sanity check: large error
        means the correspondences disagree with each other."""
        projected = self.to_world(self.pixel_points)
        return float(np.linalg.norm(projected - self.world_points, axis=1).mean())
