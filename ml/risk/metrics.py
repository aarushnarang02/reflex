"""Surrogate safety measures: the math of "how close was that?"

Two complementary questions, borrowed from decades of traffic safety
research:

TTC (time to collision) — if both users keep doing exactly what they're
doing, how many seconds until they occupy the same space? Catches
head-on and rear-end style conflicts where paths point at each other.

PET (post encroachment time) — how many seconds apart did two users
pass through the same patch of ground? Catches crossing conflicts where
paths intersect but never point at each other.

All math happens on the calibrated ground plane, in meters and seconds.
Users are modeled as circles sized by what they are.
"""

import math

import numpy as np

# Effective radius (m) of each road user class on the ground plane
USER_RADIUS = {
    "pedestrian": 0.3,
    "bicycle": 0.5,
    "motorcycle": 0.5,
    "car": 1.0,
    "bus": 1.6,
    "truck": 1.3,
}

# Interactions involving unprotected humans are weighted heavier
VULNERABILITY = {
    frozenset(): 1.0,
    frozenset({"pedestrian"}): 1.5,
    frozenset({"bicycle"}): 1.3,
    frozenset({"motorcycle"}): 1.3,
    frozenset({"pedestrian", "bicycle"}): 1.5,
    frozenset({"pedestrian", "motorcycle"}): 1.5,
    frozenset({"bicycle", "motorcycle"}): 1.3,
}

TTC_HORIZON = 3.0     # seconds — beyond this, a projected collision is routine traffic
PET_MAX = 2.0         # seconds — occupancy gaps larger than this are unremarkable
PET_RADIUS = 1.0      # meters — how close two ground points count as "same spot"
SPEED_NORM = 50.0     # km/h closing speed that saturates the severity factor


def time_to_collision(p1, v1, r1, p2, v2, r2) -> float:
    """Seconds until two constant-velocity circles touch. inf if never.

    Solves |(p2-p1) + (v2-v1) t| = r1+r2 for the smallest positive t.
    """
    r = np.asarray(p2, float) - np.asarray(p1, float)
    v = np.asarray(v2, float) - np.asarray(v1, float)
    radius = r1 + r2

    dist2 = float(r @ r)
    if dist2 <= radius * radius:
        return 0.0  # already overlapping

    speed2 = float(v @ v)
    if speed2 < 1e-12:
        return math.inf  # no relative motion

    rv = float(r @ v)
    if rv >= 0:
        return math.inf  # moving apart

    disc = rv * rv - speed2 * (dist2 - radius * radius)
    if disc < 0:
        return math.inf  # closest approach still misses
    return (-rv - math.sqrt(disc)) / speed2


def closing_speed_kmh(v1, v2) -> float:
    """How fast the gap between two users is shrinking, in km/h."""
    v = np.asarray(v1, float) - np.asarray(v2, float)
    return float(np.linalg.norm(v)) * 3.6


def post_encroachment_time(times_a, pos_a, times_b, pos_b) -> float:
    """Smallest time gap between the two users occupying the same ground
    patch (within PET_RADIUS). inf if their paths never overlap."""
    if len(times_a) == 0 or len(times_b) == 0:
        return math.inf
    pa = np.asarray(pos_a, float)
    pb = np.asarray(pos_b, float)
    ta = np.asarray(times_a, float)
    tb = np.asarray(times_b, float)
    # pairwise distances between every point of A and every point of B
    d = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=2)
    close = d < PET_RADIUS
    if not close.any():
        return math.inf
    gaps = np.abs(ta[:, None] - tb[None, :])[close]
    return float(gaps.min())


def vulnerability_weight(class_a: str, class_b: str) -> float:
    key = frozenset({class_a, class_b}) & {"pedestrian", "bicycle", "motorcycle"}
    return VULNERABILITY.get(key, 1.0)


def risk_score(ttc: float, closing_kmh: float, class_a: str, class_b: str,
               pet: float = math.inf) -> float:
    """Composite 0-100 danger score for one interaction moment.

    Urgency (how imminent) scaled by severity (how fast they're closing)
    scaled by vulnerability (who would get hurt). PET contributes as an
    alternative urgency signal for crossing conflicts.
    """
    urgency_ttc = max(0.0, 1.0 - ttc / TTC_HORIZON)
    urgency_pet = max(0.0, 1.0 - pet / PET_MAX)
    urgency = max(urgency_ttc, urgency_pet)
    severity = 0.4 + 0.6 * min(1.0, closing_kmh / SPEED_NORM)
    return min(100.0, 100.0 * urgency * severity * vulnerability_weight(class_a, class_b))
