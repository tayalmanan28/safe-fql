"""Plot roll-out trajectories from a trained SafeFQL_Base or SafeIFQL agent on BoatRobot.

Supports configurable N (rejection sampling budget) at evaluation time.

Usage:
    python plot_trajectories_boat_safefql_base.py results/BoatRobot/<experiment_name>
    python plot_trajectories_boat_safefql_base.py results/BoatRobot/<experiment_name> --N=1
    python plot_trajectories_boat_safefql_base.py results/BoatRobot/<experiment_name> --N=8 --extract_method=minqc
"""
import argparse
import json
import os
import re
import sys

sys.path.append(".")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from ml_collections import ConfigDict

from env.boat_robot import BoatRobot
from jaxrl5.agents import SafeFQL_Base, SafeIFQL


def to_config_dict(d):
    if isinstance(d, dict):
        return ConfigDict({k: to_config_dict(v) for k, v in d.items()})
    return d


def find_checkpoint(model_location, checkpoint=None):
    pickle_files = [f for f in os.listdir(model_location) if f.endswith(".pickle")]
    numbers = {}
    for filename in pickle_files:
        match = re.search(r"\d+", filename)
        if match:
            numbers[int(match.group())] = os.path.join(model_location, filename)

    if not numbers:
        raise ValueError(f"No checkpoints found in: {model_location}")

    if checkpoint is None:
        return numbers[max(numbers.keys())]
    if checkpoint not in numbers:
        raise ValueError(
            f"Checkpoint {checkpoint} not found. Available: {sorted(numbers.keys())}"
        )
    return numbers[checkpoint]


def build_init_states(env, csv_path, num_trajectories=None):
    data = np.genfromtxt(csv_path, delimiter=",", names=True)

    if data.size == 0:
        raise ValueError(f"CSV has no rows: {csv_path}")

    names = list(data.dtype.names or [])
    lower_to_orig = {n.lower(): n for n in names}

    if "x" in lower_to_orig and "y" in lower_to_orig:
        x_col = lower_to_orig["x"]
        y_col = lower_to_orig["y"]
        init_states = np.stack([data[x_col], data[y_col]], axis=-1).astype(np.float32)
    else:
        raw = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
        if raw.ndim == 1:
            raw = raw[None, :]
        if raw.shape[1] < 2:
            raise ValueError(f"CSV must have at least two columns for x,y: {csv_path}")
        init_states = raw[:, :2].astype(np.float32)

    if num_trajectories is not None:
        init_states = init_states[:num_trajectories]

    low, high = env.observation_space.low, env.observation_space.high
    valid = np.logical_and(init_states >= low, init_states <= high).all(axis=1)
    if not np.all(valid):
        dropped = int((~valid).sum())
        print(f"Warning: dropping {dropped} out-of-bounds initial states from CSV")
        init_states = init_states[valid]

    if init_states.shape[0] == 0:
        raise ValueError("No valid initial states available after CSV filtering")

    return init_states


def collect_trajectory(agent, env, init_state):
    """Roll out a single trajectory using agent.eval_actions()."""
    obs = env.reset(state=init_state)
    positions = [env.state[:2].copy()]
    total_reward = 0.0
    total_cost = 0.0

    for _ in range(env._max_episode_steps):
        action, agent = agent.eval_actions(obs)
        obs, reward, done, info = env.step(action)
        positions.append(env.state[:2].copy())
        total_reward += float(reward)
        total_cost += float(info.get("cost", info.get("violation", 0.0)))
        if done:
            break

    return np.asarray(positions), total_reward, total_cost, agent


def main():
    parser = argparse.ArgumentParser(
        description="Plot BoatRobot trajectories for SafeFQL_Base / SafeIFQL agents."
    )
    parser.add_argument(
        "model_location",
        type=str,
        help="Path containing config.json and model*.pickle",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=None,
        help="Override N (rejection sampling budget). Default: use config value.",
    )
    parser.add_argument(
        "--extract_method",
        type=str,
        default=None,
        help="Override extract_method (minqc / maxq / safe_maxq). Default: use config value.",
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=None,
        help="Optional cap on number of trajectories to roll out from CSV",
    )
    parser.add_argument(
        "--traj_csv",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "Traj_points.csv"),
        help="CSV with initial trajectory points (expects X,Y columns)",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        default=None,
        help="Checkpoint number to load (default: latest)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Output figure DPI",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Output image filename (default: trajectories_boat_N{N}.pdf)",
    )
    args = parser.parse_args()

    # ---- Load config ----
    with open(os.path.join(args.model_location, "config.json"), "r") as f:
        cfg = to_config_dict(json.load(f))

    env = BoatRobot(id=0, seed=0)

    config_dict = dict(cfg["agent_kwargs"])
    model_cls_name = config_dict.pop("model_cls")
    model_cls = {"SafeFQL_Base": SafeFQL_Base, "SafeIFQL": SafeIFQL}[model_cls_name]
    config_dict.pop("cost_scale", None)
    config_dict["env_max_steps"] = env._max_episode_steps

    agent = model_cls.create(
        cfg["seed"], env.observation_space, env.action_space, **config_dict
    )

    ckpt_path = find_checkpoint(args.model_location, args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    agent = agent.load(ckpt_path)

    # ---- Override N and extract_method if specified ----
    eval_N = args.N if args.N is not None else agent.N
    agent = agent.replace(N=eval_N)
    if args.extract_method is not None:
        agent = agent.replace(extract_method=args.extract_method)

    print(f"Model: {model_cls_name}, N={agent.N}, extract_method={agent.extract_method}")

    # ---- Build initial states ----
    init_states = build_init_states(
        env,
        csv_path=args.traj_csv,
        num_trajectories=args.num_trajectories,
    )
    print(f"Rolling out {len(init_states)} trajectories on BoatRobot...")

    # ---- Collect trajectories ----
    fig, ax = plt.subplots(figsize=(8, 6))
    ax = env.plot_task(ax)
    cmap = plt.cm.viridis

    returns = []
    costs = []
    safe_count = 0
    for idx, s0 in enumerate(init_states):
        print(
            f"[{idx + 1:03d}/{len(init_states):03d}] "
            f"start_state=({s0[0]:.3f}, {s0[1]:.3f})"
        )
        traj, ret, cost, agent = collect_trajectory(agent, env, s0)
        color = cmap(idx / max(1, len(init_states) - 1))
        ax.plot(traj[:, 0], traj[:, 1], "-", color=color, linewidth=1.4, alpha=0.82)
        ax.plot(traj[0, 0], traj[0, 1], "o", color=color, markersize=4, alpha=0.85)
        returns.append(ret)
        costs.append(cost)
        if cost == 0.0:
            safe_count += 1
        print(
            f"[{idx + 1:03d}/{len(init_states):03d}] "
            f"done steps={len(traj) - 1:03d} return={ret:.4f} cost={cost:.4f}"
        )

    # ---- Plot formatting ----
    x_lo, y_lo = env.observation_space.low
    x_hi, y_hi = env.observation_space.high
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_yticks([-2, -1, 0, 1, 2])

    # River flow drag arrows
    arrow_xs = np.linspace(x_lo + 0.3, x_hi - 0.3, 8)
    arrow_ys = np.linspace(y_lo + 0.2, y_hi - 0.2, 9)
    ax_grid, ay_grid = np.meshgrid(arrow_xs, arrow_ys)
    drift = 2.0 - 0.5 * ay_grid ** 2
    drift = np.clip(drift, 0, None)
    ax.quiver(
        ax_grid, ay_grid,
        drift, np.zeros_like(drift),
        color="lightskyblue", alpha=0.45,
        scale=25, width=0.006, headwidth=3.5, headlength=4,
        zorder=0,
    )

    # ---- Save ----
    out_dir = os.path.join(args.model_location, "imgs")
    os.makedirs(out_dir, exist_ok=True)
    if args.output_name is None:
        output_name = f"trajectories_boat_N{eval_N}.pdf"
    else:
        output_name = args.output_name
    save_path = os.path.join(out_dir, output_name)
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    safety_rate = 100.0 * safe_count / len(init_states)
    print(f"\nSaved plot to {save_path}")
    print(
        f"Stats: "
        f"mean_return={np.mean(returns):.4f}, "
        f"mean_cost={np.mean(costs):.4f}, "
        f"max_cost={np.max(costs):.4f}, "
        f"safety_rate={safety_rate:.1f}% ({safe_count}/{len(init_states)})"
    )


if __name__ == "__main__":
    main()
