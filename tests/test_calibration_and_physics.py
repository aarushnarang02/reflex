"""Physics components verified against hand computed answers."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.calibration.homography import GroundPlane
from ml.tracking.trajectory import Trajectory


def test_homography_recovers_known_scale():
    # A 100px square maps to a 5m square: 1px = 5cm everywhere
    plane = GroundPlane(
        pixel_points=np.array([[0, 0], [100, 0], [100, 100], [0, 100]]),
        world_points=np.array([[0, 0], [5, 0], [5, 5], [0, 5]]),
    )
    center = plane.to_world([[50, 50]])[0]
    assert center == pytest.approx([2.5, 2.5], abs=1e-6)
    assert plane.reprojection_error() < 1e-6


def test_homography_perspective_case():
    # Trapezoid in the image (far edge appears shorter) mapping to a
    # 4m x 10m rectangle — classic road viewed from a pole camera.
    plane = GroundPlane(
        pixel_points=np.array([[100, 300], [500, 300], [350, 100], [250, 100]]),
        world_points=np.array([[0, 0], [4, 0], [4, 10], [0, 10]]),
    )
    # Midpoint of the near edge in pixels must land at the middle of the
    # near edge in world coords.
    near_mid = plane.to_world([[300, 300]])[0]
    assert near_mid == pytest.approx([2.0, 0.0], abs=1e-6)
    # Equal image distances are not equal world distances: nearby road
    # fills more pixels per meter, so the image's vertical midpoint sits
    # SHORT of the world midline (y=5m).
    world_of_pixel_middle = plane.to_world([[300, 200]])[0]
    assert 0.0 < world_of_pixel_middle[1] < 5.0


def test_kalman_converges_to_true_velocity():
    # A user walking a perfectly straight line at 1.4 m/s (typical
    # pedestrian). After a couple of seconds the filter's velocity
    # estimate must land on the truth.
    traj = Trajectory(track_id=1, class_name="pedestrian")
    dt = 1 / 30
    for i in range(90):  # 3 seconds
        traj.update(t=i * dt, x=1.4 * i * dt, y=0.0)
    assert traj.speed_mps == pytest.approx(1.4, abs=0.05)
    assert traj.heading_deg == pytest.approx(0.0, abs=2.0)


def test_kalman_smooths_measurement_noise():
    # Same walker, but detections jitter by ±20cm of gaussian noise.
    # Raw frame-to-frame differences would imply absurd speeds
    # (0.2m in 1/30s is 6 m/s); the filter must see through it.
    rng = np.random.default_rng(7)
    traj = Trajectory(track_id=2, class_name="pedestrian")
    dt = 1 / 30
    for i in range(150):
        noise = rng.normal(0, 0.2, size=2)
        traj.update(t=i * dt, x=1.4 * i * dt + noise[0], y=noise[1])
    assert traj.speed_mps == pytest.approx(1.4, abs=0.25)


def test_stationary_user_reads_near_zero_speed():
    rng = np.random.default_rng(3)
    traj = Trajectory(track_id=3, class_name="car")
    for i in range(120):
        noise = rng.normal(0, 0.1, size=2)
        traj.update(t=i / 30, x=5.0 + noise[0], y=5.0 + noise[1])
    assert traj.speed_mps < 0.3
