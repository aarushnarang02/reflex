"""Trajectory physics: from noisy ground plane positions to smooth motion.

Raw detections jitter frame to frame — a box edge shifting by a few pixels
can fake a burst of speed. Each track therefore runs through a Kalman
filter with a constant velocity model: every frame it predicts where the
user should be, measures where they appear to be, and blends the two in
proportion to how much it trusts each. Out the other side come smooth
positions and, crucially, velocities in meters per second.
"""

from dataclasses import dataclass, field

import numpy as np

# Filter tuning: how much acceleration we allow between frames (process
# noise) vs how noisy we believe detections are (measurement noise).
ACCEL_SIGMA = 3.0   # m/s^2 — road users can brake/turn hard
MEAS_SIGMA = 0.35   # m — detection box jitter projected to ground


class KalmanConstantVelocity:
    """State: [x, y, vx, vy] on the ground plane, in meters and m/s."""

    def __init__(self, x: float, y: float):
        self.state = np.array([x, y, 0.0, 0.0])
        self.cov = np.diag([MEAS_SIGMA**2, MEAS_SIGMA**2, 25.0, 25.0])
        self._H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self._R = np.eye(2) * MEAS_SIGMA**2

    def step(self, x: float, y: float, dt: float) -> None:
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        # white-noise acceleration model
        q = ACCEL_SIGMA**2
        G = np.array([[dt**2 / 2, 0], [0, dt**2 / 2], [dt, 0], [0, dt]])
        Q = G @ G.T * q

        # predict
        self.state = F @ self.state
        self.cov = F @ self.cov @ F.T + Q

        # update
        z = np.array([x, y])
        y_res = z - self._H @ self.state
        S = self._H @ self.cov @ self._H.T + self._R
        K = self.cov @ self._H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y_res
        self.cov = (np.eye(4) - K @ self._H) @ self.cov

    @property
    def position(self) -> tuple[float, float]:
        return float(self.state[0]), float(self.state[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.state[2]), float(self.state[3])


@dataclass
class Trajectory:
    """One road user's motion history on the ground plane."""

    track_id: int
    class_name: str
    times: list[float] = field(default_factory=list)
    positions: list[tuple[float, float]] = field(default_factory=list)   # smoothed
    velocities: list[tuple[float, float]] = field(default_factory=list)  # m/s
    _filter: KalmanConstantVelocity | None = None

    def update(self, t: float, x: float, y: float) -> None:
        if self._filter is None:
            self._filter = KalmanConstantVelocity(x, y)
        else:
            self._filter.step(x, y, t - self.times[-1])
        self.times.append(t)
        self.positions.append(self._filter.position)
        self.velocities.append(self._filter.velocity)

    @property
    def speed_mps(self) -> float:
        """Current smoothed speed in meters per second."""
        vx, vy = self.velocities[-1]
        return float(np.hypot(vx, vy))

    @property
    def heading_deg(self) -> float:
        vx, vy = self.velocities[-1]
        return float(np.degrees(np.arctan2(vy, vx)))

    @property
    def duration(self) -> float:
        return self.times[-1] - self.times[0] if len(self.times) > 1 else 0.0

    def stats(self) -> dict:
        speeds = np.hypot(*np.array(self.velocities[min(5, len(self.velocities) - 1):]).T) \
            if len(self.velocities) > 1 else np.array([0.0])
        return {
            "track_id": self.track_id,
            "class": self.class_name,
            "duration_s": round(self.duration, 1),
            "avg_speed_kmh": round(float(speeds.mean()) * 3.6, 1),
            "max_speed_kmh": round(float(speeds.max()) * 3.6, 1),
        }


class TrajectoryBuilder:
    """Feeds tracked detections through per-track Kalman filters."""

    def __init__(self, ground_plane):
        self.ground_plane = ground_plane
        self.trajectories: dict[int, Trajectory] = {}

    def update(self, frame_result, t: float) -> None:
        for user in frame_result.users:
            world = self.ground_plane.to_world([user.bottom_center])[0]
            traj = self.trajectories.get(user.track_id)
            if traj is None:
                traj = Trajectory(user.track_id, user.class_name)
                self.trajectories[user.track_id] = traj
            traj.update(t, float(world[0]), float(world[1]))
