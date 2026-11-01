"""Regenerate the report's figures and results: python scripts/make_figures.py

Writes docs/figures/*.png and docs/results.json. Besides the demo run, it
compares the cost used here (speed counted along the course) with an
ablation that counts the raw speed, over 20 random seeds.
"""
import contextlib
import io
import json
import math
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mppi_control import mppi_control as m

FIGURES = ROOT / "docs" / "figures"
SEEDS = range(20)
m.show_animation = False
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "savefig.dpi": 200, "savefig.bbox": "tight"})


def raw_speed_costs(states, course, obstacles):
    """Ablation: the earlier cost, which counted the raw speed."""
    gap = m.obstacle_gaps(states, obstacles)
    intrusion = np.clip(m.CLEARANCE_MARGIN - gap, 0.0, None)
    cost = (m.W_COURSE * m.squared_distance_to_course(states[..., :2], course)
            + m.W_SPEED * (states[..., 3] - m.TARGET_SPEED) ** 2
            + m.W_CLEARANCE * np.sum(intrusion ** 2, axis=-1)
            + m.COLLISION_COST * np.any(gap < 0.0, axis=-1))
    return cost.sum(axis=1)


ESS: list[float] = []  # effective sample size of every update of the last run
_weights = m.mppi_weights


def recording_weights(costs, temperature):
    weights = _weights(costs, temperature)
    ESS.append(float(1.0 / np.sum(weights ** 2)))
    return weights


m.mppi_weights = recording_weights


def run(seed):
    ESS.clear()
    with contextlib.redirect_stdout(io.StringIO()):
        return m.main(seed)


def metrics(states, controls, course, obstacles):
    distance = np.sqrt(m.squared_distance_to_course(states[:, :2], course))
    gap = m.obstacle_gaps(states, obstacles).min(axis=1)
    clear = gap > 10.0
    finished = len(controls) * m.DT < m.MAX_TIME - 1e-9
    return {"finished": bool(finished), "time": len(controls) * m.DT,
            "min_gap": float(gap.min()),
            "course_distance_clear": float(distance[clear].max()) if clear.any() else None,
            "median_ess": float(np.median(ESS))}


def seed_study():
    """The cost and control-cost weight used here, and the ablations of each
    and of both: the raw speed in the cost, and the control cost weighted by
    the temperature, as first derived."""
    results = {}
    original_costs, original_gamma = m.trajectory_costs, m.GAMMA
    variants = (("along_course", original_costs, original_gamma),
                ("raw_speed", raw_speed_costs, original_gamma),
                ("gamma_equals_lambda", original_costs, m.LAMBDA),
                ("raw_speed_gamma_equals_lambda", raw_speed_costs, m.LAMBDA))
    for name, costs, gamma in variants:
        m.trajectory_costs, m.GAMMA = costs, gamma
        results[name] = [{"seed": seed, **metrics(*run(seed))} for seed in SEEDS]
    m.trajectory_costs, m.GAMMA = original_costs, original_gamma
    return results


def figure_course(states, course, obstacles):
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    ax.plot(course[:, 0], course[:, 1], "--", color="0.5", lw=1, label="course")
    for x, y, r in obstacles:
        ax.add_patch(Circle((x, y), r, color="tab:red"))
        ax.add_patch(Circle((x, y), r + m.CAR_RADIUS + m.CLEARANCE_MARGIN,
                            fill=False, ls=":", color="tab:red", lw=0.8))
    ax.plot(states[:, 0], states[:, 1], "-", color="tab:blue", lw=1.6, label="driven path")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(FIGURES / "fig1_course.png")
    plt.close(fig)


def figure_samples(seed=3):
    """One control step near the first obstacle: the sampled trajectories
    shaded by weight, and the updated plan."""
    rng = np.random.default_rng(seed)
    course = m.make_course()
    obstacles = m.make_obstacles(course)
    state = np.array([course[0, 0], course[0, 1],
                      math.atan2(course[1, 1] - course[0, 1], course[1, 0] - course[0, 0]),
                      0.0, 0.0])
    plan = np.zeros((m.HORIZON, 2))
    progress = 0
    while math.hypot(*(m.car_center(state) - obstacles[0, :2])) > 9.0:
        local = course[max(progress - 5, 0):progress + m.COURSE_WINDOW]
        plan, _ = m.mppi_control(state, plan, local, obstacles, rng)
        state = m.motion_model(state, plan[0])
        plan = np.vstack([plan[1:], plan[-1:]])
        progress = m.nearest_index(course, state[:2], progress)
    local = course[max(progress - 5, 0):progress + m.COURSE_WINDOW]
    noise = rng.normal(0.0, m.NOISE_STD, size=(m.N_SAMPLES, m.HORIZON, 2))
    samples = m.rollout(state, plan + noise)
    costs = m.trajectory_costs(samples, local, obstacles)
    costs += m.GAMMA * np.einsum("tc,ktc->k", plan / m.NOISE_STD ** 2, noise)
    weights = m.mppi_weights(costs, m.LAMBDA)
    new_plan = np.clip(plan + np.einsum("k,ktc->tc", weights, noise),
                       -m.CONTROL_LIMITS, m.CONTROL_LIMITS)
    planned = m.rollout(state, new_plan[None])[0]
    fig, ax = plt.subplots(figsize=(6.6, 2.9))
    order = np.argsort(weights)
    shade = weights[order] / weights.max()
    for k, s in zip(order, shade):
        ax.plot(samples[k, :, 0], samples[k, :, 1], "-", lw=0.6,
                color=plt.cm.Greys(0.15 + 0.85 * s))
    ax.plot(course[:, 0], course[:, 1], "--", color="0.5", lw=1)
    for x, y, r in obstacles:
        ax.add_patch(Circle((x, y), r, color="tab:red"))
    ax.plot(planned[:, 0], planned[:, 1], "-", color="tab:green", lw=2, label="weighted plan")
    ax.plot(*state[:2], "o", color="tab:blue", label="car (rear axle)")
    ax.set_xlim(state[0] - 3, state[0] + 20)
    ax.set_ylim(state[1] - 7, state[1] + 7)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(loc="lower left", fontsize=8)
    fig.savefig(FIGURES / "fig2_samples.png")
    plt.close(fig)
    ess = 1.0 / np.sum(weights ** 2)
    return {"effective_sample_size": float(ess), "samples": m.N_SAMPLES}


def figure_profiles(states, controls):
    t = np.arange(len(states)) * m.DT
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.3))
    a1.plot(t, states[:, 3], color="tab:blue")
    a1.axhline(m.TARGET_SPEED, ls="--", color="0.5", lw=1)
    a1.set_xlabel("time [s]")
    a1.set_ylabel("speed [m/s]")
    a2.plot(t, np.degrees(states[:, 4]), color="tab:purple")
    a2.set_xlabel("time [s]")
    a2.set_ylabel("steering angle [deg]")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_profiles.png")
    plt.close(fig)


def figure_seeds(study):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.6))
    for name, color, marker, label in (
            ("raw_speed_gamma_equals_lambda", "tab:red", "x", r"raw speed, $\gamma=\lambda$"),
            ("gamma_equals_lambda", "tab:green", "s", r"along course, $\gamma=\lambda$"),
            ("raw_speed", "tab:orange", "^", r"raw speed, $\gamma=5$"),
            ("along_course", "tab:blue", "o", r"along course, $\gamma=5$ (used)")):
        rows = study[name]
        seeds = [r["seed"] for r in rows]
        a1.plot(seeds, [r["time"] for r in rows], marker, color=color, ms=4, label=label)
        a2.plot(seeds, [r["min_gap"] for r in rows], marker, color=color, ms=4, label=label)
    a1.axhline(m.MAX_TIME, ls=":", color="0.4", lw=1)
    a1.set_xlabel("seed")
    a1.set_ylabel("time to finish [s]")
    a2.axhline(0.0, ls=":", color="tab:red", lw=1)
    a2.set_xlabel("seed")
    a2.set_ylabel("closest approach [m]")
    fig.legend(*a1.get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(FIGURES / "fig4_seeds.png")
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    states, controls, course, obstacles = run(3)
    runtime = time.perf_counter() - t0
    demo = metrics(states, controls, course, obstacles)  # before ESS is reused
    figure_course(states, course, obstacles)
    sample_info = figure_samples()
    figure_profiles(states, controls)
    study = seed_study()
    figure_seeds(study)
    summary = {}
    for name, rows in study.items():
        done = [r for r in rows if r["finished"]]
        summary[name] = {
            "finished": f"{len(done)}/{len(rows)}",
            "worst_time": max(r["time"] for r in done) if done else None,
            "mean_time": float(np.mean([r["time"] for r in done])) if done else None,
            "worst_gap": min(r["min_gap"] for r in rows),
            "worst_course_distance": max(r["course_distance_clear"] for r in done
                                         if r["course_distance_clear"] is not None),
            "median_ess": float(np.median([r["median_ess"] for r in rows])),
        }
    results = {"demo_seed_3": {**demo,
                               "steps": len(controls),
                               "max_steer_deg": float(np.degrees(np.abs(states[:, 4]).max())),
                               "max_steer_rate_deg_s": float(np.degrees(np.abs(np.diff(states[:, 4])).max() / m.DT)),
                               "runtime_s": runtime,
                               "ms_per_step": 1000.0 * runtime / len(controls)},
               "sample_snapshot": sample_info,
               "seed_study": summary, "seed_rows": study}
    (ROOT / "docs" / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: results[k] for k in ("demo_seed_3", "sample_snapshot", "seed_study")}, indent=2))


if __name__ == "__main__":
    main()
