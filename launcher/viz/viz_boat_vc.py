"""Visualise the learned safety value function V_c for BoatRobot.

Usage:
    python launcher/viz/viz_boat_vc.py \
        --model_location results/BoatRobot/<experiment_name>
"""
import os
import sys
sys.path.append(".")
import re
import json
import numpy as np
from absl import app, flags
from ml_collections import ConfigDict
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Circle
import jax

from env.boat_robot import BoatRobot
from jaxrl5.agents import SafeFQL_Base, SafeFQL


FLAGS = flags.FLAGS
flags.DEFINE_string("model_location", "", "Path to the saved model directory")
flags.DEFINE_integer("resolution", 301, "Grid resolution for the value map")
flags.DEFINE_integer("dpi", 300, "DPI for the saved figure")


# =====================================================================
# Helpers
# =====================================================================

def to_config_dict(d):
    if isinstance(d, dict):
        return ConfigDict({k: to_config_dict(v) for k, v in d.items()})
    return d


label_size = 18
ticks_size = 16
width = 0.5

font = {
    "family": "Times New Roman",
    "weight": "normal",
    "size": label_size,
}
plt.rc("font", **font)


# =====================================================================
# Plotting
# =====================================================================

def plot_vc_map(ax, agent, env, resolution=301, cb=False, vmin=None, vmax=None):
    """Plot the learned V_c(s) over the (x, y) state space of BoatRobot."""
    x_lo, y_lo = env.observation_space.low
    x_hi, y_hi = env.observation_space.high

    xs = np.linspace(x_lo, x_hi, resolution)
    ys = np.linspace(y_lo, y_hi, resolution)
    x_grid, y_grid = np.meshgrid(xs, ys)

    # BoatRobot obs is just (x, y)
    batch_obs = np.stack([x_grid.ravel(), y_grid.ravel()], axis=-1).astype(
        np.float32
    )

    vc_vals = agent.safe_value.apply_fn(
        {"params": agent.safe_value.params}, jax.device_put(batch_obs)
    )
    vc_grid = np.asarray(vc_vals).reshape(x_grid.shape)

    if vmin is None:
        vmin = vc_grid.min()
    if vmax is None:
        vmax = vc_grid.max()

    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    ct = ax.contourf(
        x_grid, y_grid, vc_grid,
        norm=norm,
        levels=40,
        cmap="RdYlGn_r",  # red = high cost (unsafe), green = low cost (safe)
    )

    # Zero level-set: boundary between safe and unsafe
    ct_line = ax.contour(
        x_grid, y_grid, vc_grid,
        levels=[0], colors="blue",
        linewidths=2.0, linestyles="solid",
    )
    ax.clabel(ct_line, inline=True, fontsize=13, fmt="0")

    if cb:
        cbar = plt.colorbar(ct, ax=ax, shrink=0.85, pad=0.03)
        cbar.ax.tick_params(labelsize=ticks_size)
        cbar.set_label(r"$V_c(s)$", fontsize=label_size)

    return ax, vc_grid


def plot_env_overlay(ax, env):
    """Draw hazards and goal on top of the value map."""
    for hazard_pos in env.hazard_position_list:
        circle = Circle(
            (hazard_pos[0], hazard_pos[1]),
            hazard_pos[2],
            fill=False,
            edgecolor="red",
            linewidth=2.0,
            linestyle="--",
            label="Hazard",
        )
        ax.add_patch(circle)

    goal_circle = Circle(
        (env.goal_position[0], env.goal_position[1]),
        0.1,
        fill=True,
        alpha=0.7,
        color="lime",
        label="Goal",
    )
    ax.add_patch(goal_circle)

    return ax


def make_figure(env, agent, model_location, resolution=301, dpi=300):
    """Create and save a figure with V_c heatmap and env overlay."""
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)

    ax, vc_grid = plot_vc_map(ax, agent, env, resolution=resolution, cb=True)
    ax = plot_env_overlay(ax, env)

    x_lo, y_lo = env.observation_space.low
    x_hi, y_hi = env.observation_space.high
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlabel("x", fontsize=label_size)
    ax.set_ylabel("y", fontsize=label_size)
    ax.tick_params(labelsize=ticks_size)
    ax.set_yticks([-2, -1, 0, 1, 2])
    # ax.set_title(r"Learned Safety Value $V_c(s)$ — BoatRobot", fontsize=label_size)

    # Remove duplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=14, loc="upper right")

    img_dir = os.path.join(model_location, "imgs")
    os.makedirs(img_dir, exist_ok=True)
    save_path = os.path.join(img_dir, "viz_boat_vc.pdf")
    plt.savefig(save_path, dpi=dpi)
    print(f"Saved figure to {save_path}")

    # Also print some stats
    print(f"\nV_c statistics over ({resolution}x{resolution}) grid:")
    print(f"  min  = {vc_grid.min():.4f}")
    print(f"  max  = {vc_grid.max():.4f}")
    print(f"  mean = {vc_grid.mean():.4f}")
    frac_unsafe = (vc_grid > 0).mean()
    print(f"  frac V_c > 0 (unsafe) = {frac_unsafe:.4f}")


# =====================================================================
# Model loading
# =====================================================================

def load_model(model_location):
    with open(os.path.join(model_location, "config.json"), "r") as f:
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

    # Find latest checkpoint
    pickle_files = [f for f in os.listdir(model_location) if f.endswith(".pickle")]
    numbers = {}
    for f in pickle_files:
        match = re.search(r"\d+", f)
        if match:
            numbers[int(match.group())] = os.path.join(model_location, f)
    ckpt_path = numbers[max(numbers.keys())]
    print(f"Loading checkpoint: {ckpt_path}")
    agent = agent.load(ckpt_path)

    return env, agent


# =====================================================================
# Main
# =====================================================================

def main(_):
    env, agent = load_model(FLAGS.model_location)
    make_figure(
        env, agent, FLAGS.model_location,
        resolution=FLAGS.resolution,
        dpi=FLAGS.dpi,
    )


if __name__ == "__main__":
    app.run(main)
