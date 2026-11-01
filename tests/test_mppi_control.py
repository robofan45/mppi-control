"""
MPPI control: the car finishes the course without touching an obstacle,
stays near the course away from the obstacles, and respects its limits.

MPPI is random, and floating-point differences between platforms change a
run the way a different seed does, so every bound below holds with margin
for all of seeds 0 to 99 (the worst value over those seeds is noted).
"""
import numpy as np
import pytest

from mppi_control import mppi_control as m


@pytest.fixture(scope="module")
def run():
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(m, "show_animation", False)
        return m.main()


def test_car_finishes_the_course(run):
    states, controls, course, _ = run
    assert len(controls) * m.DT < 25.0  # worst 16.6 s
    assert np.linalg.norm(states[-1, :2] - course[-1]) < 1.0  # worst 0.30 m


def test_car_never_touches_an_obstacle(run):
    states, _, _, obstacles = run
    assert m.obstacle_gaps(states, obstacles).min() > 0.3  # worst 0.79 m


def test_car_stays_near_the_course_away_from_obstacles(run):
    states, _, course, obstacles = run
    distance = np.sqrt(m.squared_distance_to_course(states[:, :2], course))
    clear = m.obstacle_gaps(states, obstacles).min(axis=1) > 10.0
    assert clear.sum() > 50  # worst 65 steps
    assert distance[clear].max() < 2.0  # worst 0.56 m


def test_steering_acceleration_and_speed_stay_within_limits(run):
    states, controls, _, _ = run
    steer = states[:, 4]
    assert np.abs(steer).max() <= m.MAX_STEER
    assert np.abs(np.diff(steer)).max() <= m.MAX_STEER_RATE * m.DT + 1e-9
    assert np.abs(controls[:, 0]).max() <= m.MAX_ACCEL
    # only the speed along the course is held at TARGET_SPEED
    assert states[:, 3].max() < 1.1 * m.TARGET_SPEED  # worst 6.32 m/s


@pytest.mark.parametrize("seed", [0, 9])
def test_car_does_not_circle_next_to_an_obstacle(seed, monkeypatch):
    """With these seeds an earlier cost, which counted the raw speed, let
    the car circle beside the first obstacle until time ran out."""
    monkeypatch.setattr(m, "show_animation", False)
    states, controls, _, obstacles = m.main(seed)
    assert len(controls) * m.DT < 25.0
    assert m.obstacle_gaps(states, obstacles).min() > 0.3


def test_only_speed_along_the_course_counts():
    """Driving across the course at the target speed must cost as much as
    standing still; otherwise circling at full speed looks like progress."""
    course = np.column_stack([np.arange(0.0, 10.0, m.COURSE_DS),
                              np.zeros(20)])
    no_obstacles = np.array([[100.0, 100.0, 1.0]])
    along, across, standing = (
        np.array([[[5.0, 0.0, yaw, speed, 0.0]]])
        for yaw, speed in [(0.0, m.TARGET_SPEED), (np.pi / 2, m.TARGET_SPEED),
                           (0.0, 0.0)])
    costs = m.trajectory_costs(np.concatenate([along, across, standing]),
                               course, no_obstacles)
    assert costs[0] == pytest.approx(0.0, abs=1e-9)
    assert costs[1] == pytest.approx(costs[2])
    assert costs[2] == pytest.approx(m.W_SPEED * m.TARGET_SPEED ** 2)


def test_squared_distance_to_course_matches_brute_force():
    rng = np.random.default_rng(0)
    points = rng.uniform(-5.0, 80.0, size=(7, 9, 2))
    course = m.make_course()
    brute = np.min(np.sum((points[..., None, :] - course) ** 2, axis=-1),
                   axis=-1)
    np.testing.assert_allclose(m.squared_distance_to_course(points, course),
                               brute, atol=1e-8)


def test_weights_favour_cheap_samples():
    costs = np.array([3.0, 1.0, 2.0, 10.0])
    weights = m.mppi_weights(costs, temperature=1.0)
    assert weights.sum() == pytest.approx(1.0)
    assert list(np.argsort(-weights)) == [1, 2, 0, 3]
    # only cost differences matter
    np.testing.assert_allclose(m.mppi_weights(costs + 1e6, 1.0), weights)
    # a cold temperature, or costs too large for exp(), keep the best only
    np.testing.assert_allclose(m.mppi_weights(costs, 1e-3), [0, 1, 0, 0],
                               atol=1e-12)
    np.testing.assert_allclose(m.mppi_weights(costs * 1e4, 1.0),
                               [0, 1, 0, 0], atol=1e-12)


def test_update_does_not_hinge_on_one_random_draw(run):
    """Repeat MPPI updates from states of the run, with the commands the car
    went on to apply as the plan: different random draws give nearly the
    same command. If one sample took nearly all the weight, each update
    would copy that sample's noise and every seed would drive its own
    path."""
    states, controls, course, obstacles = run
    spread = []
    for i in (20, 60, 100):
        plan = controls[i:i + m.HORIZON]
        first = [m.mppi_control(states[i], plan, course, obstacles,
                                np.random.default_rng(seed))[0][0]
                 for seed in range(10)]
        spread.append(np.std(first, axis=0) / m.NOISE_STD)
    # 0.11 here; copying one sample's noise gives about 0.6
    assert np.median(spread) < 0.3


def test_same_result_twice(run, monkeypatch):
    monkeypatch.setattr(m, "show_animation", False)
    for first, second in zip(run, m.main()):
        np.testing.assert_array_equal(first, second)

