"""Plot roll-out trajectories using the *weighted BC flow* policy from a
trained SafeFQL agent on BoatRobot.

Unlike `plot_trajectories_boat.py` (which uses the one-step distilled flow),
this script generates actions by running the multi-step *BC flow model*
(trained with feasibility-weighted flow matching) via Euler integration,
then selecting among N candidates using the Q/Q_c critics.

Usage:
    python plot_trajectories_boat_bcflow.py results/BoatRobot/<experiment_name>
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
from functools import partial
from ml_collections import ConfigDict

import jax
import jax.numpy as jnp

from env.boat_robot import BoatRobot
from jaxrl5.agents import SafeFQL_Base, SafeFQL


# =====================================================================
# Helpers
# =====================================================================

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


# =====================================================================
# BC flow action selection (jitted)
# =====================================================================

@partial(jax.jit, static_argnames=("bc_flow_fn", "flow_steps"))
def _run_bc_flow(bc_flow_fn, bc_flow_params, observations, noises, flow_steps):
    """Run the multi-step BC flow model via Euler integration."""
    actions = noises
    for i in range(flow_steps):
        t = jnp.full((*observations.shape[:-1], 1), i / flow_steps)
        vels = bc_flow_fn({"params": bc_flow_params}, observations, actions, t)
        actions = actions + vels / flow_steps
    return jnp.clip(actions, -1, 1)


@partial(jax.jit, static_argnames=("critic_fn",))
def _compute_q(critic_fn, critic_params, observations, actions):
    q_values = critic_fn({"params": critic_params}, observations, actions)
    return q_values.min(axis=0)


@partial(jax.jit, static_argnames=("safe_critic_fn",))
def _compute_qc(safe_critic_fn, safe_critic_params, observations, actions):
    qc_values = safe_critic_fn({"params": safe_critic_params}, observations, actions)
    return qc_values.max(axis=0)


def eval_actions_bcflow(agent, observations, extract_method="safe_maxq"):
    """Generate N action candidates via the *multi-step BC flow*, then
    select by safety (Q_c) and reward (Q) using the specified extract_method.

    Returns:
        action: np.ndarray of shape (act_dim,)
        agent: updated agent (with new rng)
    """
    rng = agent.rng
    assert len(observations.shape) == 1

    observations = jax.device_put(observations)
    obs_repeated = jnp.expand_dims(observations, axis=0).repeat(agent.N, axis=0)

    rng, noise_key = jax.random.split(rng)
    noises = jax.random.normal(noise_key, (agent.N, agent.act_dim))

    # Multi-step BC flow inference (instead of one-step flow)
    actions = _run_bc_flow(
        agent.actor_bc_flow.apply_fn,
        agent.actor_bc_flow.params,
        obs_repeated,
        noises,
        agent.flow_steps,
    )

    # Evaluate candidates with critics
    qs = _compute_q(
        agent.target_critic.apply_fn, agent.target_critic.params,
        obs_repeated, actions,
    )
    qcs = _compute_qc(
        agent.safe_target_critic.apply_fn, agent.safe_target_critic.params,
        obs_repeated, actions,
    )

    if agent.critic_type == "qc":
        qcs = qcs - agent.qc_thres

    # Action selection
    if extract_method == "maxq":
        idx = jnp.argmax(qs)
    elif extract_method == "minqc":
        idx = jnp.argmin(qcs)
    elif extract_method == "safe_maxq":
        safe_mask = qcs <= 0
        has_safe = jnp.any(safe_mask)
        safe_qs = jnp.where(safe_mask, qs, -jnp.inf)
        safe_idx = jnp.argmax(safe_qs)
        unsafe_idx = jnp.argmin(qcs)
        idx = jnp.where(has_safe, safe_idx, unsafe_idx)
    else:
        raise ValueError(f"Invalid extract_method: {extract_method}")

    action = actions[idx]
    return np.array(action.squeeze()), agent.replace(rng=rng)


# =====================================================================
# Trajectory collection
# =====================================================================

def collect_trajectory(agent, env, init_state, extract_method="safe_maxq"):
    obs = env.reset(state=init_state)
    positions = [env.state[:2].copy()]
    total_reward = 0.0
    total_cost = 0.0

    for _ in range(env._max_episode_steps):
        action, agent = eval_actions_bcflow(agent, obs, extract_method=extract_method)
        obs, reward, done, info = env.step(action)
        positions.append(env.state[:2].copy())
        total_reward += float(reward)
        total_cost += float(info.get("cost", info.get("violation", 0.0)))
        if done:
            break

    return np.asarray(positions), total_reward, total_cost, agent


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot BoatRobot trajectories using the weighted BC flow policy."
    )
    parser.add_argument(
        "model_location", type=str,
        help="Path containing config.json and model*.pickle",
    )
    parser.add_argument(
        "--num_trajectories", type=int, default=None,
        help="Optional cap on number of trajectories to roll out from CSV",
    )
    parser.add_argument(
        "--traj_csv", type=str,
        default=os.path.join(os.path.dirname(__file__), "Traj_points.csv"),
        help="CSV with initial trajectory points (expects X,Y columns)",
    )
    parser.add_argument(
        "--checkpoint", type=int, default=None,
        help="Checkpoint number to load (default: latest)",
    )
    parser.add_argument(
        "--extract_method", type=str, default=None,
        help="Action selection method: maxq, minqc, safe_maxq (default: from config)",
    )
    parser.add_argument(
        "--N", type=int, default=None,
        help="Override number of action candidates (default: from config)",
    )
    parser.add_argument(
        "--dpi", type=int, default=220,
        help="Output figure DPI",
    )
    parser.add_argument(
        "--output_name", type=str, default="trajectories_boat_bcflow.png",
        help="Output image filename",
    )
    args = parser.parse_args()

    # Load config
    with open(os.path.join(args.model_location, "config.json"), "r") as f:
        cfg = to_config_dict(json.load(f))

    env = BoatRobot(id=0, seed=0)

    config_dict = dict(cfg["agent_kwargs"])
    model_cls_name = config_dict.pop("model_cls")
    model_cls = {"SafeFQL_Base": SafeFQL_Base, "SafeFQL": SafeFQL}[model_cls_name]
    config_dict.pop("cost_scale", None)
    config_dict["env_max_steps"] = env._max_episode_steps

    agent = model_cls.create(
        cfg["seed"], env.observation_space, env.action_space, **config_dict
    )

    ckpt_path = find_checkpoint(args.model_location, args.checkpoint)
    print(f"Loading checkpoint: {ckpt_path}")
    agent = agent.load(ckpt_path)

    # Override N if requested
    if args.N is not None:
        print(f"Overriding N: {agent.N} -> {args.N}")
        agent = agent.replace(N=args.N)

    # Determine extract method
    extract_method = args.extract_method or agent.extract_method
    print(f"Using extract_method: {extract_method}")
    print(f"Using N={agent.N} BC flow candidates per step")

    # Load initial states
    init_states = build_init_states(
        env, csv_path=args.traj_csv, num_trajectories=args.num_trajectories,
    )
    print(f"Rolling out {len(init_states)} trajectories on BoatRobot (BC flow)...")

    # Collect trajectories
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
        traj, ret, cost, agent = collect_trajectory(
            agent, env, s0, extract_method=extract_method
        )
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
    ax.set_title(
        f"{model_cls_name} BC-Flow - BoatRobot Trajectories (n={len(init_states)})\n"
        f"mean_return={np.mean(returns):.2f}, mean_cost={np.mean(costs):.2f}, "
        f"extract={extract_method}, N={agent.N}"
    )

    out_dir = os.path.join(args.model_location, "imgs")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, args.output_name)
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved plot to {save_path}")
    print(
        "Stats: "
        f"mean_return={np.mean(returns):.4f}, "
        f"mean_cost={np.mean(costs):.4f}, "
        f"max_cost={np.max(costs):.4f}"
    )


if __name__ == "__main__":
    main()
