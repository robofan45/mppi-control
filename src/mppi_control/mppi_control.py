"""

Model Predictive Path Integral (MPPI) control

author: Mahim Trivedi

A car follows a course and steers around obstacles placed on it. Every
control step MPPI samples many random variations of its planned acceleration
and steering-rate sequence, simulates the car for each one, and scores the
trajectories: stay near the course, keep the target speed along the course,
keep clear of the obstacles. The plan then moves toward the low-cost samples, each weighted by
exp(-cost / lambda). The first command of the plan is applied and the plan
shifts one step, ready for the next control step.

The steering angle is part of the car's state and MPPI samples its rate of
change, as a real steering system works: the samples explore sustained turns,
and the wheel never jumps between control steps.

Reference:
- Williams, Aldrich and Theodorou, "Model Predictive Path Integral Control:
  From Theory to Parallel Computation", Journal of Guidance, Control, and
  Dynamics, 2017.
- Williams et al., "Information Theoretic MPC for Model-Based Reinforcement
  Learning", ICRA 2017.
- Williams et al., "Information-Theoretic Model Predictive Control: Theory
  and Applications to Autonomous Driving", IEEE Transactions on Robotics,
  2018.

"""
import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.interpolate import CubicSpline

# vehicle
WB = 2.0  # [m] wheelbase
MAX_STEER = np.radians(35.0)  # [rad]
MAX_STEER_RATE = np.radians(40.0)  # [rad/s]
MAX_ACCEL = 3.0  # [m/s^2]
CAR_LENGTH, CAR_WIDTH = 3.0, 1.4  # [m] body, centred on the wheelbase
CAR_RADIUS = 1.7  # [m] circle around the body centre that covers the body

# MPPI
DT = 0.1  # [s] control period
HORIZON = 25  # [steps] look-ahead of the plan (2.5 s)
N_SAMPLES = 300  # sampled control sequences per control step
LAMBDA = 20.0  # temperature: smaller trusts the best samples more
GAMMA = 5.0  # weight of the control cost, well below LAMBDA (see mppi_control)
NOISE_STD = np.array([1.0, np.radians(20.0)])  # (accel, steer rate) noise
CONTROL_LIMITS = np.array([MAX_ACCEL, MAX_STEER_RATE])

# cost
TARGET_SPEED = 6.0  # [m/s] along the course
W_COURSE = 1.0  # squared distance to the course
W_SPEED = 1.0  # squared error of the speed along the course
W_CLEARANCE = 200.0  # squared intrusion into the clearance margin
CLEARANCE_MARGIN = 1.0  # [m] extra gap kept from obstacles
COLLISION_COST = 1e4  # per time step a sample spends inside an obstacle

COURSE_DS = 0.5  # [m] spacing of the course points
COURSE_WINDOW = 60  # course points ahead of the car used by the cost
MAX_TIME = 40.0  # [s]

show_animation = True


def make_course() -> np.ndarray:
    """Course points (N, 2), COURSE_DS apart along a smooth S-shaped road: a
    natural cubic spline through the waypoints, parameterised by the
    distance travelled between them."""
    waypoints = np.array([[0.0, 0.0], [15.0, 4.0], [30.0, 0.0],
                          [45.0, -4.0], [60.0, 0.0], [75.0, 3.0]])
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(waypoints, axis=0).T))])
    spline = CubicSpline(s, waypoints, bc_type="natural")
    return spline(np.arange(0.0, s[-1], COURSE_DS))


def make_obstacles(course: np.ndarray) -> np.ndarray:
    """Obstacles (x, y, radius) on the course, slightly off its centre line
    so the car has a preferred side to pass on."""
    obstacles = []
    for s, offset, radius in [(22.0, 0.4, 1.0), (52.0, -0.5, 1.2)]:
        i = int(s / COURSE_DS)
        direction = course[i + 1] - course[i]
        normal = np.array([-direction[1], direction[0]]) / np.hypot(*direction)
        x, y = course[i] + offset * normal
        obstacles.append((x, y, radius))
    return np.array(obstacles)


def motion_model(state: np.ndarray, control: np.ndarray) -> np.ndarray:
    """One DT step of a kinematic bicycle model, for arrays of states
    (..., 5) = (x, y, yaw, v, steering angle) at the rear axle and controls
    (..., 2) = (acceleration, steering rate)."""
    x, y, yaw, v, steer = (state[..., i] for i in range(5))
    accel = np.clip(control[..., 0], -MAX_ACCEL, MAX_ACCEL)
    steer_rate = np.clip(control[..., 1], -MAX_STEER_RATE, MAX_STEER_RATE)
    steer = np.clip(steer + steer_rate * DT, -MAX_STEER, MAX_STEER)
    return np.stack([x + v * np.cos(yaw) * DT,
                     y + v * np.sin(yaw) * DT,
                     yaw + v / WB * np.tan(steer) * DT,
                     v + accel * DT,
                     steer], axis=-1)


def car_center(states: np.ndarray) -> np.ndarray:
    """Body centre (..., 2): half a wheelbase ahead of the rear axle."""
    yaw = states[..., 2]
    return states[..., :2] + 0.5 * WB * np.stack([np.cos(yaw), np.sin(yaw)],
                                                 axis=-1)


def rollout(state: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """States (K, T, 5) reached by applying the control sequences
    (K, T, 2) from one start state (5,)."""
    states = np.empty(controls.shape[:2] + state.shape)
    current = np.broadcast_to(state, controls.shape[:1] + state.shape)
    for t in range(controls.shape[1]):
        current = motion_model(current, controls[:, t])
        states[:, t] = current
    return states


def obstacle_gaps(states: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
    """Free distance (..., M) between the car's covering circle and each
    obstacle; negative means a collision."""
    center = car_center(states)
    return (np.linalg.norm(center[..., None, :] - obstacles[:, :2], axis=-1)
            - obstacles[:, 2] - CAR_RADIUS)


def nearest_course_points(xy: np.ndarray, course: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Index of the nearest course point (...) for each point (..., 2), and
    the squared distance to it, from |p - c|^2 = |p|^2 - 2 p.c + |c|^2 with
    one matrix product."""
    points = xy.reshape(-1, 2)
    dist_sq = (np.sum(points ** 2, axis=1)[:, None] - 2.0 * points @ course.T
               + np.sum(course ** 2, axis=1))
    nearest = np.argmin(dist_sq, axis=1)
    closest = np.clip(dist_sq[np.arange(len(points)), nearest], 0.0, None)
    return nearest.reshape(xy.shape[:-1]), closest.reshape(xy.shape[:-1])


def squared_distance_to_course(xy: np.ndarray,
                               course: np.ndarray) -> np.ndarray:
    """Squared distance (...) from each point (..., 2) to the course."""
    return nearest_course_points(xy, course)[1]


def course_headings(course: np.ndarray) -> np.ndarray:
    """Direction of travel at each course point."""
    step = np.diff(course, axis=0)
    heading = np.arctan2(step[:, 1], step[:, 0])
    return np.append(heading, heading[-1])


def trajectory_costs(states: np.ndarray, course: np.ndarray,
                     obstacles: np.ndarray) -> np.ndarray:
    """Cost (K,) of each trajectory in states (K, T, 5). The speed counts
    only along the course: otherwise circling next to an obstacle at full
    speed, crossing the course again and again, is cheaper than driving
    around the obstacle."""
    nearest, course_dist_sq = nearest_course_points(states[..., :2], course)
    along = states[..., 3] * np.cos(states[..., 2]
                                    - course_headings(course)[nearest])
    gap = obstacle_gaps(states, obstacles)
    intrusion = np.clip(CLEARANCE_MARGIN - gap, 0.0, None)
    cost = (W_COURSE * course_dist_sq
            + W_SPEED * (along - TARGET_SPEED) ** 2
            + W_CLEARANCE * np.sum(intrusion ** 2, axis=-1)
            + COLLISION_COST * np.any(gap < 0.0, axis=-1))
    return cost.sum(axis=1)


def mppi_weights(costs: np.ndarray, temperature: float) -> np.ndarray:
    """Weights exp(-cost / temperature), normalised to sum to one. The
    lowest cost is subtracted first, so the best sample gets exp(0) = 1
    and the sum cannot underflow to zero."""
    weights = np.exp(-(costs - costs.min()) / temperature)
    return weights / weights.sum()


def mppi_control(state: np.ndarray, plan: np.ndarray, course: np.ndarray,
                 obstacles: np.ndarray, rng: np.random.Generator
                 ) -> tuple[np.ndarray, np.ndarray]:
    """One MPPI update of the plan (T, 2). Also returns the sampled
    trajectories (K, T, 5), for plotting."""
    noise = rng.normal(0.0, NOISE_STD, size=(N_SAMPLES, HORIZON, 2))
    samples = rollout(state, plan + noise)
    costs = trajectory_costs(samples, course, obstacles)
    # control cost gamma * u_t^T Sigma^-1 noise_t, summed over the horizon.
    # With gamma = lambda, as first derived, the weights treat the samples
    # as if drawn around zero control: a plan several noise widths from
    # zero is pushed back to zero every step, and one sample takes nearly
    # all the weight. Williams et al. 2018 use gamma = lambda (1 - alpha).
    costs += GAMMA * np.einsum("tc,ktc->k", plan / NOISE_STD ** 2, noise)
    weights = mppi_weights(costs, LAMBDA)
    plan = plan + np.einsum("k,ktc->tc", weights, noise)
    return np.clip(plan, -CONTROL_LIMITS, CONTROL_LIMITS), samples


def nearest_index(course: np.ndarray, position: np.ndarray,
                  start: int) -> int:
    """Index of the course point nearest to position, searched forward from
    start so the car's progress along the course never jumps back."""
    window = course[start:start + COURSE_WINDOW]
    return start + int(np.argmin(np.sum((window - position) ** 2, axis=1)))


def plot_car(state: np.ndarray, color: str = "k") -> None:  # pragma: no cover
    """Outline of the car body at its pose."""
    x, y, yaw = state[:3]
    rear, front = 0.5 * (WB - CAR_LENGTH), 0.5 * (WB + CAR_LENGTH)
    half = 0.5 * CAR_WIDTH
    body = np.array([[rear, -half], [front, -half], [front, half],
                     [rear, half], [rear, -half]])
    rot = np.array([[math.cos(yaw), -math.sin(yaw)],
                    [math.sin(yaw), math.cos(yaw)]])
    outline = body @ rot.T + [x, y]
    plt.fill(outline[:, 0], outline[:, 1], color=color, alpha=0.8)


def plot_scene(course: np.ndarray, obstacles: np.ndarray,
               driven: np.ndarray) -> None:  # pragma: no cover
    plt.plot(course[:, 0], course[:, 1], "--", color="0.55", label="course")
    for x, y, radius in obstacles:
        plt.gca().add_patch(Circle((x, y), radius, color="tab:red"))
    plt.plot(driven[:, 0], driven[:, 1], "-b", lw=2, label="driven path")


def plot_frame(state: np.ndarray, samples: np.ndarray, plan: np.ndarray,
               course: np.ndarray, obstacles: np.ndarray,
               driven: np.ndarray, time: float) -> None:  # pragma: no cover
    plt.cla()
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if getattr(event, "key", None) == 'escape'
                       else None])
    plt.plot(samples[::5, :, 0].T, samples[::5, :, 1].T, "-", color="0.8",
             lw=0.6)
    plot_scene(course, obstacles, driven)
    planned = rollout(state, plan[None])[0]
    plt.plot(planned[:, 0], planned[:, 1], "-g", lw=2, label="MPPI plan")
    plot_car(state)
    plt.axis((state[0] - 10.0, state[0] + 22.0, state[1] - 10.0,
              state[1] + 10.0))
    plt.gca().set_aspect("equal")
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.title(f"MPPI  t = {time:4.1f} s   speed = {state[3] * 3.6:4.1f} km/h"
              "\ngrey: sampled trajectories")
    plt.pause(0.001)


def main(seed: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray]:
    print(__file__ + " start!!")
    rng = np.random.default_rng(seed)
    course = make_course()
    obstacles = make_obstacles(course)
    start_yaw = math.atan2(course[1, 1] - course[0, 1],
                           course[1, 0] - course[0, 0])
    state = np.array([course[0, 0], course[0, 1], start_yaw, 0.0, 0.0])
    plan: np.ndarray = np.zeros((HORIZON, 2))
    states, controls = [state], []
    progress, time = 0, 0.0

    while time < MAX_TIME and progress < len(course) - 1:
        local = course[max(progress - 5, 0):progress + COURSE_WINDOW]
        plan, samples = mppi_control(state, plan, local, obstacles, rng)
        state = motion_model(state, plan[0])
        controls.append(plan[0])
        plan = np.vstack([plan[1:], plan[-1:]])  # shift, repeat the last
        states.append(state)
        progress = nearest_index(course, state[:2], progress)
        time += DT

        if show_animation:  # pragma: no cover
            plot_frame(state, samples, plan, course, obstacles,
                       np.array(states), time)

    if show_animation:  # pragma: no cover
        plt.cla()
        plot_scene(course, obstacles, np.array(states))
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title(f"MPPI: course finished in {time:.1f} s")
        plt.show()

    return np.array(states), np.array(controls), course, obstacles


if __name__ == '__main__':
    main()
