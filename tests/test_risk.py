"""Risk math verified against hand computed scenarios."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml.risk import metrics
from ml.risk.engine import RiskEngine
from ml.tracking.trajectory import Trajectory


def test_ttc_head_on():
    # Two cars 10m apart driving straight at each other at 5 m/s each.
    # Gap shrinks at 10 m/s; they touch when centers are 2.0m apart
    # (two 1.0m radii): (10 - 2) / 10 = 0.8 seconds.
    ttc = metrics.time_to_collision(
        (0, 0), (5, 0), 1.0,
        (10, 0), (-5, 0), 1.0)
    assert ttc == pytest.approx(0.8, abs=1e-9)


def test_ttc_parallel_traffic_never_collides():
    # Two cars side by side in adjacent lanes, same speed.
    ttc = metrics.time_to_collision(
        (0, 0), (10, 0), 1.0,
        (0, 3.5), (10, 0), 1.0)
    assert math.isinf(ttc)


def test_ttc_near_miss_geometry():
    # Head-on but offset by 2.5m laterally: paths pass close but the
    # circles (sum of radii 2.0m) never touch.
    ttc = metrics.time_to_collision(
        (0, 0), (5, 0), 1.0,
        (10, 2.5), (-5, 0), 1.0)
    assert math.isinf(ttc)


def test_ttc_diverging():
    ttc = metrics.time_to_collision(
        (0, 0), (-5, 0), 1.0,
        (10, 0), (5, 0), 1.0)
    assert math.isinf(ttc)


def test_pet_crossing_paths():
    # A passes through the origin at t=1.0; B passes the same spot at
    # t=1.8. Post encroachment time = 0.8 seconds.
    times_a = [0.0, 0.5, 1.0, 1.5]
    pos_a = [(-10, 0), (-5, 0), (0, 0), (5, 0)]
    times_b = [0.8, 1.3, 1.8, 2.3]
    pos_b = [(0, -10), (0, -5), (0, 0), (0, 5)]
    pet = metrics.post_encroachment_time(times_a, pos_a, times_b, pos_b)
    assert pet == pytest.approx(0.8, abs=1e-9)


def test_pet_paths_never_cross():
    pet = metrics.post_encroachment_time(
        [0, 1], [(0, 0), (10, 0)],
        [0, 1], [(0, 50), (10, 50)])
    assert math.isinf(pet)


def test_risk_score_weights_pedestrians_heavier():
    # identical geometry, different participants
    car_car = metrics.risk_score(ttc=1.0, closing_kmh=30, class_a="car", class_b="car")
    car_ped = metrics.risk_score(ttc=1.0, closing_kmh=30, class_a="car", class_b="pedestrian")
    assert car_ped == pytest.approx(car_car * 1.5, rel=1e-6)


def test_risk_score_zero_when_no_urgency():
    assert metrics.risk_score(ttc=math.inf, closing_kmh=100,
                              class_a="car", class_b="car") == 0.0
    assert metrics.risk_score(ttc=10.0, closing_kmh=100,
                              class_a="car", class_b="car") == 0.0


def _make_traj(track_id, cls, path, dt=0.1):
    traj = Trajectory(track_id=track_id, class_name=cls)
    for i, (x, y) in enumerate(path):
        traj.update(t=i * dt, x=x, y=y)
    return traj


def test_engine_flags_head_on_conflict():
    # Car and cyclist driving straight at each other: car heading right
    # at 5 m/s, cyclist heading left at 4 m/s, closing at 9 m/s. The
    # engine watches the whole approach, frame by frame.
    dt = 0.1
    car = Trajectory(track_id=1, class_name="car")
    bike = Trajectory(track_id=2, class_name="bicycle")
    engine = RiskEngine()
    for i in range(15):
        t = i * dt
        car.update(t=t, x=5.0 * t, y=0.0)
        bike.update(t=t, x=20.0 - 4.0 * t, y=0.0)
        engine.update({1: car, 2: bike}, t)
    engine.finish({1: car, 2: bike})
    assert engine.events, "closing head-on pair must produce an event"
    ev = engine.events[0]
    assert ev.peak_score >= 30
    assert ev.class_b == "bicycle" or ev.class_a == "bicycle"


def test_engine_suppresses_companions():
    # A person walking their bicycle: two detections 0.3m apart moving
    # in lockstep. Overlapping circles give TTC=0 (looks maximally
    # urgent), but the companion filter must recognize one entity.
    dt = 0.1
    ped = Trajectory(track_id=1, class_name="pedestrian")
    bike = Trajectory(track_id=2, class_name="bicycle")
    engine = RiskEngine()
    for i in range(30):
        t = i * dt
        ped.update(t=t, x=1.4 * t, y=0.0)
        bike.update(t=t, x=1.4 * t, y=0.3)
        engine.update({1: ped, 2: bike}, t)
    engine.finish({1: ped, 2: bike})
    assert not engine.events


def test_engine_suppresses_following_traffic():
    # Two cars queued at a light, 2.2m gap, creeping forward together at
    # ~1 m/s. Their circles nearly touch (TTC ~ 0) but there is no energy
    # in it — this is traffic, not a near miss.
    dt = 0.1
    lead = Trajectory(track_id=1, class_name="car")
    tail = Trajectory(track_id=2, class_name="car")
    engine = RiskEngine()
    for i in range(40):
        t = i * dt
        lead.update(t=t, x=2.2 + 1.0 * t, y=0.0)
        tail.update(t=t, x=1.0 * t, y=0.0)
        engine.update({1: lead, 2: tail}, t)
    engine.finish({1: lead, 2: tail})
    assert not engine.events


def test_engine_keeps_hard_rear_end_approach():
    # A car bearing down on a stopped car at 8 m/s (~29 km/h closing).
    # Same lane, same heading — but this one has real energy and must
    # NOT be dismissed as following traffic.
    dt = 0.1
    stopped = Trajectory(track_id=1, class_name="car")
    charger = Trajectory(track_id=2, class_name="car")
    engine = RiskEngine()
    for i in range(15):
        t = i * dt
        stopped.update(t=t, x=14.0, y=0.0)
        charger.update(t=t, x=8.0 * t, y=0.0)
        engine.update({1: stopped, 2: charger}, t)
    engine.finish({1: stopped, 2: charger})
    assert engine.events, "high-energy rear-end approach must stay flagged"


def test_engine_skips_pedestrian_pairs_by_default():
    # Two people walking straight at each other — not a traffic conflict.
    dt = 0.1
    a = Trajectory(track_id=1, class_name="pedestrian")
    b = Trajectory(track_id=2, class_name="pedestrian")
    engine = RiskEngine()
    for i in range(30):
        t = i * dt
        a.update(t=t, x=1.4 * t, y=0.0)
        b.update(t=t, x=6.0 - 1.4 * t, y=0.0)
        engine.update({1: a, 2: b}, t)
    engine.finish({1: a, 2: b})
    assert not engine.events
    assert not engine.interaction_log  # pair never even scored

    # ...unless a crowd deployment opts in
    a2 = Trajectory(track_id=1, class_name="pedestrian")
    b2 = Trajectory(track_id=2, class_name="pedestrian")
    engine2 = RiskEngine(score_pedestrian_pairs=True)
    for i in range(30):
        t = i * dt
        a2.update(t=t, x=1.4 * t, y=0.0)
        b2.update(t=t, x=6.0 - 1.4 * t, y=0.0)
        engine2.update({1: a2, 2: b2}, t)
    assert engine2.interaction_log


def test_engine_ignores_calm_traffic():
    # Two cars far apart moving the same direction.
    car1 = _make_traj(1, "car", [(i * 0.5, 0) for i in range(30)])
    car2 = _make_traj(2, "car", [(i * 0.5, 30) for i in range(30)])
    engine = RiskEngine()
    engine.update({1: car1, 2: car2}, t=3.0)
    engine.finish({1: car1, 2: car2})
    assert not engine.events
