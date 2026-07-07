"""The risk engine: watches every pair of road users and flags near misses.

Every frame, each pair of simultaneously present road users gets scored.
When a pair's score crosses the event threshold, an event opens; while it
stays open the engine records the worst moment (peak score, minimum TTC,
minimum distance); when risk subsides or a user leaves the scene, the
event closes, PET is computed over the full trajectories, and the event
is finalized.
"""

import math
from dataclasses import dataclass, field

import numpy as np

from ml.risk import metrics

EVENT_THRESHOLD = 30.0   # score that opens an event
EVENT_COOLDOWN = 1.0     # seconds below threshold before an event closes
MAX_PAIR_DISTANCE = 15.0 # meters — pairs farther apart are not evaluated

# Companion suppression: a rider and their bicycle, or one vehicle double
# detected, look like a permanent "near miss". The tell: genuine conflicts
# approach, peak, and separate; companions stay glued together for nearly
# their whole shared screen time.
COMPANION_DISTANCE = 2.0  # meters counted as "together"
COMPANION_FRACTION = 0.7  # together for >70% of co-presence = same entity
MIN_EVENT_SAMPLES = 5     # scored moments required before an event is credible

# Following-traffic suppression: queued and platooning vehicles sit close
# together with near-zero TTC, but there is almost no kinetic energy in
# the interaction. A vehicle-vehicle pair only counts as a conflict if
# their gap ever closed with real speed (crossing paths, hard braking).
VEHICLES = frozenset({"car", "truck", "bus"})
MIN_VEHICLE_CLOSING_KMH = 15.0


@dataclass
class NearMissEvent:
    track_a: int
    track_b: int
    class_a: str
    class_b: str
    t_start: float
    t_end: float = 0.0
    t_peak: float = 0.0
    peak_score: float = 0.0
    min_ttc: float = math.inf
    min_distance: float = math.inf
    max_closing_kmh: float = 0.0
    pet: float = math.inf
    peak_location: tuple[float, float] = (0.0, 0.0)
    closed: bool = False


@dataclass
class _PairState:
    event: NearMissEvent | None = None
    below_since: float | None = None


class RiskEngine:
    def __init__(self, threshold: float = EVENT_THRESHOLD,
                 score_pedestrian_pairs: bool = False,
                 in_valid_region=None):
        self.threshold = threshold
        # pedestrian-pedestrian proximity is not a traffic conflict; off by
        # default, but a campus/crowd deployment can turn it on
        self.score_pedestrian_pairs = score_pedestrian_pairs
        # only score pairs where the calibration is trustworthy; far outside
        # the calibrated zone, compressed geometry manufactures fake conflicts
        self.in_valid_region = in_valid_region or (lambda x, y: True)
        self._pairs: dict[tuple[int, int], _PairState] = {}
        self.events: list[NearMissEvent] = []
        # every scored moment, kept so sparse footage still yields insight
        self.interaction_log: list[dict] = []

    def _suppress(self, ev: NearMissEvent) -> bool:
        """True when a flagged pair should not become an event: the tracks
        moved as one entity (rider + bike, duplicate detection), coexisted
        too briefly to trust, or are just vehicles following in traffic."""
        samples = [r for r in self.interaction_log
                   if r["a"] == ev.track_a and r["b"] == ev.track_b]
        if len(samples) < MIN_EVENT_SAMPLES:
            return True  # a blink of co-presence is detector noise, not evidence
        together = sum(r["distance"] < COMPANION_DISTANCE for r in samples)
        if together / len(samples) > COMPANION_FRACTION:
            return True
        if (ev.class_a in VEHICLES and ev.class_b in VEHICLES
                and ev.max_closing_kmh < MIN_VEHICLE_CLOSING_KMH):
            return True  # queued / platooning traffic, not a conflict
        return False

    def update(self, active: dict[int, "Trajectory"], t: float) -> list[dict]:
        """Score all co-present pairs at time t. Returns risky moments for
        overlay drawing: [{tracks, score, ttc, positions}, ...]."""
        overlays = []
        ids = sorted(active)
        for i, a_id in enumerate(ids):
            for b_id in ids[i + 1:]:
                a, b = active[a_id], active[b_id]
                if (not self.score_pedestrian_pairs
                        and a.class_name == "pedestrian"
                        and b.class_name == "pedestrian"):
                    continue
                pa, va = np.array(a.positions[-1]), np.array(a.velocities[-1])
                pb, vb = np.array(b.positions[-1]), np.array(b.velocities[-1])
                distance = float(np.linalg.norm(pa - pb))
                if distance > MAX_PAIR_DISTANCE:
                    continue
                mid = (pa + pb) / 2
                if not self.in_valid_region(float(mid[0]), float(mid[1])):
                    continue

                ttc = metrics.time_to_collision(
                    pa, va, metrics.USER_RADIUS[a.class_name],
                    pb, vb, metrics.USER_RADIUS[b.class_name],
                )
                closing = metrics.closing_speed_kmh(va, vb)
                score = metrics.risk_score(ttc, closing, a.class_name, b.class_name)

                self.interaction_log.append({
                    "t": t, "a": a_id, "b": b_id, "distance": distance,
                    "ttc": ttc, "closing_kmh": closing, "score": score,
                })
                if score > 0:
                    overlays.append({
                        "a": a_id, "b": b_id, "score": score, "ttc": ttc,
                    })
                self._advance_pair((a_id, b_id), a, b, t, score, ttc,
                                   distance, closing, pa, pb)
        self._close_stale(set(ids), t)
        return overlays

    def _advance_pair(self, key, a, b, t, score, ttc, distance, closing, pa, pb):
        state = self._pairs.setdefault(key, _PairState())
        if score >= self.threshold:
            if state.event is None:
                state.event = NearMissEvent(
                    track_a=key[0], track_b=key[1],
                    class_a=a.class_name, class_b=b.class_name, t_start=t,
                )
            state.below_since = None
            ev = state.event
            if score > ev.peak_score:
                ev.peak_score = score
                ev.t_peak = t
                ev.peak_location = tuple(((pa + pb) / 2).tolist())
            ev.min_ttc = min(ev.min_ttc, ttc)
            ev.min_distance = min(ev.min_distance, distance)
            ev.max_closing_kmh = max(ev.max_closing_kmh, closing)
            ev.t_end = t
        elif state.event is not None:
            if state.below_since is None:
                state.below_since = t
            elif t - state.below_since >= EVENT_COOLDOWN:
                self._finalize(key, a, b)

    def _close_stale(self, active_ids: set, t: float) -> None:
        for key, state in list(self._pairs.items()):
            if state.event is not None and not (key[0] in active_ids and key[1] in active_ids):
                self._finalize_by_key(key)

    def _finalize(self, key, a, b) -> None:
        state = self._pairs[key]
        ev = state.event
        state.event = None
        state.below_since = None
        if self._suppress(ev):
            return
        ev.pet = metrics.post_encroachment_time(a.times, a.positions, b.times, b.positions)
        ev.closed = True
        self.events.append(ev)

    def _finalize_by_key(self, key) -> None:
        # track left the scene; finalize with whatever we have
        state = self._pairs[key]
        ev = state.event
        state.event = None
        state.below_since = None
        if self._suppress(ev):
            return
        ev.closed = True
        self.events.append(ev)

    def finish(self, trajectories: dict) -> None:
        """End of video: close any events still open and compute their PET."""
        for key, state in self._pairs.items():
            if state.event is not None:
                ev = state.event
                state.event = None
                if self._suppress(ev):
                    continue
                a, b = trajectories.get(ev.track_a), trajectories.get(ev.track_b)
                if a and b:
                    ev.pet = metrics.post_encroachment_time(
                        a.times, a.positions, b.times, b.positions)
                ev.closed = True
                self.events.append(ev)
