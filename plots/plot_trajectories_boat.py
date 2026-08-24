"""Plot roll-out trajectories from a trained SafeFQL_Base/SafeFQL agent on BoatRobot.

Usage:
    python plot_trajectories_boat.py results/BoatRobot/<experiment_name>
"""
import argparse
from functools import partial
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
from jaxrl5.agents import SafeFQL_Base, SafeFQL
from jaxrl5.agents import NormalizedSafeFQL


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


# Jitted one-step flow for the N=1 fast path — no critic eval needed.
@partial(jax.jit, static_argnames=("onestep_fn",))
def _fast_onestep_action(onestep_fn, onestep_params, observation, noise):
    """Run one-step flow for a single observation and noise vector."""
    obs = jnp.expand_dims(observation, axis=0)
    n = jnp.expand_dims(noise, axis=0)
    action = onestep_fn({"params": onestep_params}, obs, n)
    return jnp.clip(action.squeeze(0), -1, 1)


def collect_trajectory(agent, env, init_state):
    obs = env.reset(state=init_state)
    positions = [env.state[:2].copy()]
    total_reward = 0.0
    total_cost = 0.0

    if agent.N == 1:
        # Fast path: skip critic evaluation and pytree copies entirely.
        rng = agent.rng
        onestep_fn = agent.actor_onestep_flow.apply_fn
        onestep_params = agent.actor_onestep_flow.params
        for _ in range(env._max_episode_steps):
            rng, noise_key = jax.random.split(rng)
            noise = jax.random.normal(noise_key, (agent.act_dim,))
            obs_jax = jax.device_put(obs)
            action = _fast_onestep_action(onestep_fn, onestep_params, obs_jax, noise)
            action = np.array(action)
            obs, reward, done, info = env.step(action)
            positions.append(env.state[:2].copy())
            total_reward += float(reward)
            total_cost += float(info.get("cost", info.get("violation", 0.0)))
            if done:
                break
        agent = agent.replace(rng=rng)
    else:
        # N > 1: use full eval_actions with critic-based selection.
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
        description="Plot all BoatRobot roll-out trajectories in one figure."
    )
    parser.add_argument(
        "model_location",
        type=str,
        help="Path containing config.json and model*.pickle",
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
        default="trajectories_boat.pdf",
        help="Output image filename",
    )
    args = parser.parse_args()

    with open(os.path.join(args.model_location, "config.json"), "r") as f:
        cfg = to_config_dict(json.load(f))

    env = BoatRobot(id=0, seed=0)

    config_dict = dict(cfg["agent_kwargs"])
    model_cls_name = config_dict.pop("model_cls")
    model_cls = {"SafeFQL_Base": SafeFQL_Base, "SafeFQL": SafeFQL, "NormalizedSafeFQL": NormalizedSafeFQL}[model_cls_name]
    config_dict.pop("cost_scale", None)
    config_dict["env_max_steps"] = env._max_episode_steps

    agent = model_cls.create(
        cfg["seed"], env.observation_space, env.action_space, **config_dict
    )

    ckpt_path = find_checkpoint(args.model_location, args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    agent = agent.load(ckpt_path)

    # Set the evaluation agent to use only 1 action candidate for a deterministic trajectory rollout
    agent = agent.replace(N=1)

    init_states = build_init_states(
        env,
        csv_path=args.traj_csv,
        num_trajectories=args.num_trajectories,
    )
    print(f"Rolling out {len(init_states)} trajectories on BoatRobot...")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax = env.plot_task(ax)
    cmap = plt.cm.viridis

    returns = []
    costs = []
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
        print(
            f"[{idx + 1:03d}/{len(init_states):03d}] "
            f"done steps={len(traj) - 1:03d} return={ret:.4f} cost={cost:.4f}"
        )

    x_lo, y_lo = env.observation_space.low
    x_hi, y_hi = env.observation_space.high
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_yticks([-2, -1, 0, 1, 2])

    # River flow drag arrows: drift = 2 - 0.5*y², rightward, strongest at y=0
    arrow_xs = np.linspace(x_lo + 0.3, x_hi - 0.3, 8)
    arrow_ys = np.linspace(y_lo + 0.2, y_hi - 0.2, 9)
    ax_grid, ay_grid = np.meshgrid(arrow_xs, arrow_ys)
    drift = 2.0 - 0.5 * ay_grid ** 2  # rightward drag velocity
    drift = np.clip(drift, 0, None)    # no leftward drag
    ax.quiver(
        ax_grid, ay_grid,
        drift, np.zeros_like(drift),
        color="lightskyblue", alpha=0.45,
        scale=25, width=0.006, headwidth=3.5, headlength=4,
        zorder=0,
    )

    # ax.set_title(f"{model_cls_name} - BoatRobot Trajectories (n={len(init_states)})")

    out_dir = os.path.join(args.model_location, "imgs")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, args.output_name)
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {save_path}")
    print(
        "Stats: "
        f"mean_return={np.mean(returns):.4f}, "
        f"mean_cost={np.mean(costs):.4f}, "
        f"max_cost={np.max(costs):.4f}"
    )


if __name__ == "__main__":
    main()
