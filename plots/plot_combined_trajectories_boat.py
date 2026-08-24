"""Plot combined trajectories of SafeFQL_Base, SafeIFQL, and SafeFQL on one BoatRobot figure.

Rolls out trajectories per (agent, N) combination from 2 fixed starting points.
SafeFQL_Base and SafeIFQL are rolled out at N = 1, 2, 4, 8, 16.
SafeFQL is rolled out at N = 1 only.
All trajectories from both starting points appear on a single plot.

Trajectories can be cached as CSVs to avoid re-running evaluation on repeat plots.

Usage:
    # First run: roll out trajectories and save CSVs + plot
    python plot_combined_trajectories_boat.py \
        --safefql_base_model   results/BoatRobot/<safefql_base_experiment> \
        --ifql_model    results/BoatRobot/<safefql_base_ifql_experiment> \
        --safefql_model results/BoatRobot/<safefql_experiment>

    # Subsequent runs: just re-plot from cached CSVs (no model loading)
    python plot_combined_trajectories_boat.py --from_cache
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

CACHE_DIR = os.path.join(os.path.dirname(__file__), "trajectory_cache")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__))

# N values to evaluate for SafeFQL_Base and SafeIFQL
N_VALUES = [1, 4, 8, 16]

# Fixed starting points
START_POINTS = [
    np.array([-2.7, 0.7], dtype=np.float32),
    np.array([-2.0, -1.1], dtype=np.float32),
]


# =====================================================================
# Trajectory collection (only needed when not using cache)
# =====================================================================

def load_agent(model_location, checkpoint=None):
    """Load an agent from a model directory (auto-detects SafeFQL_Base/SafeIFQL/SafeFQL)."""
    import jax
    import jax.numpy as jnp
    from ml_collections import ConfigDict
    from env.boat_robot import BoatRobot
    from jaxrl5.agents import SafeFQL_Base, SafeIFQL, SafeFQL

    def to_config_dict(d):
        if isinstance(d, ConfigDict):
            return d
        if isinstance(d, dict):
            return ConfigDict({k: to_config_dict(v) for k, v in d.items()})
        return d

    with open(os.path.join(model_location, "config.json"), "r") as f:
        cfg = to_config_dict(json.load(f))

    env = BoatRobot(id=0, seed=0)
    config_dict = dict(cfg["agent_kwargs"])
    model_cls_name = config_dict.pop("model_cls")
    model_cls = {"SafeFQL_Base": SafeFQL_Base, "SafeIFQL": SafeIFQL, "SafeFQL": SafeFQL}[model_cls_name]
    config_dict.pop("cost_scale", None)
    config_dict["env_max_steps"] = env._max_episode_steps

    agent = model_cls.create(cfg["seed"], env.observation_space, env.action_space, **config_dict)

    # Find checkpoint
    pickle_files = [f for f in os.listdir(model_location) if f.endswith(".pickle")]
    numbers = {}
    for fn in pickle_files:
        match = re.search(r"\d+", fn)
        if match:
            numbers[int(match.group())] = os.path.join(model_location, fn)
    if not numbers:
        raise ValueError(f"No checkpoints in {model_location}")
    ckpt = numbers[max(numbers.keys())] if checkpoint is None else numbers[checkpoint]
    print(f"  Loading checkpoint: {ckpt}")
    agent = agent.load(ckpt)
    return agent, env, model_cls_name


def collect_trajectory(agent, env, init_state):
    """Roll out one trajectory using agent.eval_actions()."""
    obs = env.reset(state=init_state)
    positions = [env.state[:2].copy()]
    total_cost = 0.0

    for _ in range(env._max_episode_steps):
        action, agent = agent.eval_actions(obs)
        obs, reward, done, info = env.step(action)
        positions.append(env.state[:2].copy())
        total_cost += float(info.get("cost", info.get("violation", 0.0)))
        if done:
            break

    return np.asarray(positions), total_cost, agent


def run_and_cache(model_location, agent_label, N_list, checkpoint=None):
    """Roll out trajectories for an agent at each N from all start points, save to CSV."""
    agent, env, cls_name = load_agent(model_location, checkpoint)
    os.makedirs(CACHE_DIR, exist_ok=True)
    results = {}

    for N in N_list:
        for pt_idx, start_state in enumerate(START_POINTS):
            print(f"  {agent_label} N={N} pt{pt_idx} ({start_state[0]:.1f}, {start_state[1]:.1f}) ...")
            agent_n = agent.replace(N=N)
            traj, cost, agent_n = collect_trajectory(agent_n, env, start_state)
            safe = cost == 0.0
            print(f"    steps={len(traj)-1}, cost={cost:.4f}, safe={safe}")

            csv_path = os.path.join(CACHE_DIR, f"{agent_label}_N{N}_pt{pt_idx}.csv")
            np.savetxt(csv_path, traj, delimiter=",", header="x,y", comments="")
            results[(agent_label, N, pt_idx)] = traj

    return results


# =====================================================================
# Plotting
# =====================================================================

# Visual style per agent family
AGENT_STYLES = {
    "SafeFQL_Base":       {"color": "#850f67", "linestyle": "-"},
    "SafeIFQL":  {"color": "#00897B", "linestyle": "-"},
    "Ours":        {"color": "#ff9500", "linestyle": "-"},
}

# Line width varies by N to distinguish trajectories
N_LINEWIDTHS = {1: 1.4, 4: 1.8, 8: 2.0, 16: 2.2}
N_ALPHAS     = {1: 0.55, 4: 0.75, 8: 0.85, 16: 0.95}

# SafeFQL/Ours gets extra emphasis
OURS_LINEWIDTH = 3.0
OURS_ALPHA = 0.95
OURS_ZORDER = 5


def load_cached_trajectories():
    """Load all CSVs from cache dir."""
    trajectories = {}
    if not os.path.isdir(CACHE_DIR):
        return trajectories
    for fn in sorted(os.listdir(CACHE_DIR)):
        if not fn.endswith(".csv"):
            continue
        base = fn[:-4]  # remove .csv
        # Parse: AGENT_N<digits>_pt<digits>
        match = re.match(r"(.+)_N(\d+)_pt(\d+)$", base)
        if match:
            label = match.group(1)
            N = int(match.group(2))
            pt_idx = int(match.group(3))
            traj = np.genfromtxt(os.path.join(CACHE_DIR, fn), delimiter=",", skip_header=1)
            trajectories[(label, N, pt_idx)] = traj
    return trajectories


def plot_all(trajectories, output_name="combined_trajectories_boat.pdf"):
    """Plot all trajectories on one BoatRobot figure."""
    from env.boat_robot import BoatRobot

    env = BoatRobot(id=0, seed=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = env.plot_task(ax)

    # River drift arrows
    x_lo, y_lo = env.observation_space.low
    x_hi, y_hi = env.observation_space.high
    arrow_xs = np.linspace(x_lo + 0.3, x_hi - 0.3, 8)
    arrow_ys = np.linspace(y_lo + 0.2, y_hi - 0.2, 9)
    ax_grid, ay_grid = np.meshgrid(arrow_xs, arrow_ys)
    drift = np.clip(2.0 - 0.5 * ay_grid ** 2, 0, None)
    ax.quiver(
        ax_grid, ay_grid, drift, np.zeros_like(drift),
        color="lightskyblue", alpha=0.45,
        scale=25, width=0.006, headwidth=3.5, headlength=4,
        zorder=0,
    )

    # Plot trajectories — only add legend entry once per (label, N)
    legend_added = set()
    # Sort so "Ours" is plotted last (on top)
    sorted_items = sorted(trajectories.items(), key=lambda x: (0 if x[0][0] != "Ours" else 1, x[0][0], x[0][1], x[0][2]))
    for (label, N, pt_idx), traj in sorted_items:
        style = AGENT_STYLES.get(label, {"color": "gray", "linestyle": "-"})

        # Ours gets special emphasis
        if label == "Ours":
            lw = OURS_LINEWIDTH
            alpha = OURS_ALPHA
            zorder = OURS_ZORDER
        else:
            lw = N_LINEWIDTHS.get(N, 1.5)
            alpha = N_ALPHAS.get(N, 0.7)
            zorder = 3

        legend_key = (label, N)
        leg_label = f"{label} (N={N})" if legend_key not in legend_added else None
        legend_added.add(legend_key)

        ax.plot(
            traj[:, 0], traj[:, 1],
            linestyle=style["linestyle"],
            color=style["color"],
            linewidth=lw, alpha=alpha,
            label=leg_label,
            zorder=zorder,
        )
        # Start marker
        ax.plot(traj[0, 0], traj[0, 1], "o", color=style["color"],
                markersize=6 if label == "Ours" else 5,
                alpha=0.95, zorder=zorder + 1)

    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")
    ax.set_xlabel("x", fontsize=14, fontweight="bold")
    ax.set_ylabel("y", fontsize=14, fontweight="bold")
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.tick_params(labelsize=12)

    # Legend below the plot — manually ordered: SafeFQL_Base all N, then SafeIFQL all N, then Ours
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))

    desired_order = []
    for agent_name in ["SafeFQL_Base", "SafeIFQL"]:
        for n in [1, 4, 8, 16]:
            key = f"{agent_name} (N={n})"
            if key in label_to_handle:
                desired_order.append((label_to_handle[key], key))
    # Ours last
    ours_key = "Ours (N=1)"
    if ours_key in label_to_handle:
        desired_order.append((label_to_handle[ours_key], ours_key))

    if desired_order:
        ordered_handles, ordered_labels = zip(*desired_order)
    else:
        ordered_handles, ordered_labels = handles, labels

    ax.legend(
        ordered_handles, ordered_labels,
        fontsize=15, loc="center left",
        bbox_to_anchor=(1.02, 0.5), ncol=1,
        frameon=True, columnspacing=1.0,
    )

    save_path = os.path.join(OUTPUT_DIR, output_name)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {save_path}")


# =====================================================================
# Inference time benchmarking
# =====================================================================

def benchmark_agent(agent, env, N, label, num_warmup=2, num_episodes=10):
    """Measure mean inference time (seconds) per episode (400 steps).

    For SafeFQL (Ours) at N=1, uses the direct one-step flow (no critic eval)
    since there's only one candidate and the policy is already safety-aware.
    For all other agents/N, uses the standard eval_actions path.
    """
    import time
    import jax
    import jax.numpy as jnp
    from functools import partial

    agent = agent.replace(N=N)

    # SafeFQL at N=1: use direct one-step flow (no critic overhead)
    use_fast_path = (label == "Ours" and N == 1 and hasattr(agent, "actor_onestep_flow"))

    if use_fast_path:
        @partial(jax.jit, static_argnames=("onestep_fn",))
        def _fast_action(onestep_fn, onestep_params, observation, noise):
            obs_b = jnp.expand_dims(observation, axis=0)
            n_b = jnp.expand_dims(noise, axis=0)
            action = onestep_fn({"params": onestep_params}, obs_b, n_b)
            return jnp.clip(action.squeeze(0), -1, 1)

        onestep_fn = agent.actor_onestep_flow.apply_fn
        onestep_params = agent.actor_onestep_flow.params
        rng = agent.rng

        # Warmup episodes
        for _ in range(num_warmup):
            obs = env.reset(state=START_POINTS[0])
            obs_jax = jax.device_put(np.array(obs))
            done = False
            while not done:
                rng, key = jax.random.split(rng)
                noise = jax.random.normal(key, (agent.act_dim,))
                action = _fast_action(onestep_fn, onestep_params, obs_jax, noise)
                obs, _, done, _ = env.step(np.array(action))
                obs_jax = jax.device_put(np.array(obs))

        # Timed episodes
        times = []
        for _ in range(num_episodes):
            obs = env.reset(state=START_POINTS[0])
            obs_jax = jax.device_put(np.array(obs))
            done = False
            t0 = time.perf_counter()
            while not done:
                rng, key = jax.random.split(rng)
                noise = jax.random.normal(key, (agent.act_dim,))
                action = _fast_action(onestep_fn, onestep_params, obs_jax, noise)
                obs, _, done, _ = env.step(np.array(action))
                obs_jax = jax.device_put(np.array(obs))
            t1 = time.perf_counter()
            times.append(t1 - t0)
    else:
        # Standard path: eval_actions (sampling + critic selection)
        # Warmup episodes
        for _ in range(num_warmup):
            obs = env.reset(state=START_POINTS[0])
            done = False
            while not done:
                action, agent = agent.eval_actions(obs)
                obs, _, done, _ = env.step(action)

        # Timed episodes
        times = []
        for _ in range(num_episodes):
            obs = env.reset(state=START_POINTS[0])
            done = False
            t0 = time.perf_counter()
            while not done:
                action, agent = agent.eval_actions(obs)
                obs, _, done, _ = env.step(action)
            t1 = time.perf_counter()
            times.append(t1 - t0)

    return np.mean(times), np.std(times)


def run_time_benchmark(args):
    """Benchmark inference time and produce a combined training + inference plot."""
    import matplotlib.ticker as mticker

    # ==================================================================
    # Hardcoded training times (seconds) — from benchmark_training_time.sh
    # ==================================================================
    TRAINING_TIMES = {
        "SafeFQL_Base":       113.0,
        "SafeIFQL":   64.0,
        "Ours":        128.0,
    }

    colors = {
        "SafeFQL_Base": "#850f67",
        "SafeIFQL": "#00897B",
        "Ours": "#ff9500",
    }

    # ==================================================================
    # Benchmark inference times
    # ==================================================================
    results = {}  # (label, N) -> (mean_time_s, std_time_s)

    agents_to_bench = []
    if args.safefql_base_model:
        agents_to_bench.append(("SafeFQL_Base", args.safefql_base_model, N_VALUES))
    if args.ifql_model:
        agents_to_bench.append(("SafeIFQL", args.ifql_model, N_VALUES))
    if args.safefql_model:
        agents_to_bench.append(("Ours", args.safefql_model, [1]))

    if not agents_to_bench:
        print("No model paths provided. Use --safefql_base_model, --ifql_model, --safefql_model.")
        return

    for label, model_path, n_list in agents_to_bench:
        print(f"=== Benchmarking {label} ===")
        agent, env, _ = load_agent(model_path)
        for N in n_list:
            print(f"  N={N} ...", end=" ", flush=True)
            mean_t, std_t = benchmark_agent(agent, env, N, label)
            results[(label, N)] = (mean_t, std_t)
            print(f"mean={mean_t:.3f}s ± {std_t:.3f}s per episode")

    # ==================================================================
    # Build combined figure: two subplots with independent x-axes
    # ==================================================================
    agent_order = ["SafeFQL_Base", "SafeIFQL", "Ours"]
    agent_order = [a for a in agent_order if a in [l for l, _, _ in agents_to_bench]]

    # --- Collect inference bar entries ---
    # --- Collect inference bar entries grouped by agent ---
    # Each entry: (agent_name, N, mean, std, color)
    infer_entries = []
    for agent_name in agent_order:
        n_list = N_VALUES if agent_name != "Ours" else [1]
        for N in n_list:
            if (agent_name, N) in results:
                m, s = results[(agent_name, N)]
                infer_entries.append((agent_name, N, m, s, colors[agent_name]))

    # --- Training bar entries ---
    train_bars = []
    for agent_name in agent_order:
        if agent_name in TRAINING_TIMES:
            train_bars.append((agent_name, TRAINING_TIMES[agent_name], 0, colors[agent_name]))

    n_train = len(train_bars)
    n_infer = len(infer_entries)
    bar_height = 0.55

    # Use GridSpec so training panel can be shorter than inference panel
    import matplotlib.gridspec as gridspec
    n_grid_rows = max(n_infer, 6)  # grid rows to subdivide vertically
    fig = plt.figure(figsize=(14, n_infer * 0.6 + 2))
    gs = gridspec.GridSpec(n_grid_rows, 2, figure=fig, wspace=0.25,
                           width_ratios=[1, 1])

    # Training: only spans middle portion of left column (vertically compact)
    train_start = (n_grid_rows - n_train) // 2
    train_end = train_start + n_train + 1
    ax_train = fig.add_subplot(gs[train_start:train_end, 0])

    # Inference: spans full height of right column
    ax_infer = fig.add_subplot(gs[:, 1])

    # ---- Left: Training time (seconds) ----
    train_y = np.arange(n_train)[::-1]
    train_bar_height = 0.45
    for i, (label, mean, std, color) in enumerate(train_bars):
        ax_train.barh(
            train_y[i], mean, train_bar_height,
            color=color, edgecolor="white", linewidth=0.5,
            alpha=0.9, zorder=2,
        )
        ax_train.text(mean + 1.5, train_y[i], f"{mean:.0f}s", va="center", ha="left",
                      fontsize=12, fontweight="bold", color=color)

    ax_train.set_yticks(train_y)
    ax_train.set_yticklabels([b[0] for b in train_bars], fontsize=10, fontweight="bold",
                              rotation=90, va="center")
    ax_train.set_xlabel("Training Time (seconds)", fontsize=13, fontweight="bold")
    ax_train.set_title("Training Compute Time", fontsize=15, fontweight="bold", pad=10)
    ax_train.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
    ax_train.tick_params(axis="x", labelsize=11)
    ax_train.grid(axis="x", alpha=0.2, zorder=0)
    ax_train.set_axisbelow(True)
    ax_train.set_xlim(0, max(b[1] for b in train_bars) * 1.2)
    ax_train.spines["top"].set_visible(False)
    ax_train.spines["right"].set_visible(False)
    ax_train.set_ylim(-0.6, n_train - 0.4)

    # ---- Right: Inference time (seconds per episode) ----
    # Assign y positions (top to bottom) and track agent group ranges
    infer_y = np.arange(n_infer)[::-1]
    agent_groups = {}  # agent_name -> list of y positions
    tick_labels = []

    for i, (agent_name, N, mean, std, color) in enumerate(infer_entries):
        ax_infer.barh(
            infer_y[i], mean, bar_height,
            xerr=std, capsize=3,
            color=color, edgecolor="white", linewidth=0.5,
            alpha=0.9, zorder=2,
            error_kw=dict(elinewidth=1.0, capthick=0.8, alpha=0.7),
        )
        max_std = max(e[3] for e in infer_entries)
        ax_infer.text(mean + max_std * 0.3 + 0.02,
                      infer_y[i], f"{mean:.2f}s", va="center", ha="left",
                      fontsize=10, fontweight="bold", color=color)

        tick_labels.append(f"N={N}")
        agent_groups.setdefault(agent_name, []).append(infer_y[i])

    ax_infer.set_yticks(infer_y)
    ax_infer.set_yticklabels(tick_labels, fontsize=9, fontweight="bold")

    # Add vertical agent name labels and separator lines between groups
    group_list = list(agent_groups.items())
    for g_idx, (agent_name, y_positions) in enumerate(group_list):
        mid_y = (min(y_positions) + max(y_positions)) / 2
        color = colors.get(agent_name, "#333333")
        ax_infer.text(-0.10, mid_y, agent_name,
                      transform=ax_infer.get_yaxis_transform(),
                      va="center", ha="right",
                      fontsize=10, fontweight="bold", color=color, rotation=90)

        # Horizontal separator between groups
        if g_idx < len(group_list) - 1:
            next_y_positions = group_list[g_idx + 1][1]
            sep_y = (min(y_positions) + max(next_y_positions)) / 2
            ax_infer.axhline(y=sep_y, color="gray", linewidth=0.8,
                             linestyle="--", alpha=0.4, zorder=1)

    ax_infer.set_xlabel("Inference Time per Episode (seconds)", fontsize=13, fontweight="bold")
    ax_infer.set_title("Inference Compute Time", fontsize=15, fontweight="bold", pad=10)
    ax_infer.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax_infer.tick_params(axis="x", labelsize=11)
    ax_infer.grid(axis="x", alpha=0.2, zorder=0)
    ax_infer.set_axisbelow(True)
    max_infer = max(e[2] + e[3] for e in infer_entries)
    ax_infer.set_xlim(0, max_infer * 1.35)
    ax_infer.spines["top"].set_visible(False)
    ax_infer.spines["right"].set_visible(False)

    out_path = os.path.join(OUTPUT_DIR, "compute_time_comparison.pdf")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved compute-time plot to {out_path}")


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Combined trajectory comparison: SafeFQL_Base / SafeIFQL / SafeFQL on BoatRobot"
    )
    parser.add_argument("--safefql_base_model", type=str, default=None,
                        help="Path to SafeFQL_Base model directory")
    parser.add_argument("--ifql_model", type=str, default=None,
                        help="Path to SafeIFQL model directory")
    parser.add_argument("--safefql_model", type=str, default=None,
                        help="Path to SafeFQL (SafeFQL) model directory")
    parser.add_argument("--from_cache", action="store_true",
                        help="Skip evaluation, just re-plot from cached CSVs")
    parser.add_argument("--output_name", type=str,
                        default="combined_trajectories_boat.pdf",
                        help="Output filename")
    parser.add_argument("--time_complexity", action="store_true",
                        help="Instead of trajectories, benchmark and plot inference time per action")
    args = parser.parse_args()

    # ---- Time complexity mode ----
    if args.time_complexity:
        run_time_benchmark(args)
        return

    # ---- Trajectory mode ----
    if args.from_cache:
        print("Loading trajectories from cache...")
        trajectories = load_cached_trajectories()
        if not trajectories:
            print(f"No cached CSVs found in {CACHE_DIR}. Run without --from_cache first.")
            return
        print(f"Loaded {len(trajectories)} cached trajectories")
    else:
        trajectories = {}

        if args.safefql_base_model:
            print("=== SafeFQL_Base ===")
            trajectories.update(
                run_and_cache(args.safefql_base_model, "SafeFQL_Base", N_VALUES)
            )

        if args.ifql_model:
            print("=== SafeIFQL ===")
            trajectories.update(
                run_and_cache(args.ifql_model, "SafeIFQL", N_VALUES)
            )

        if args.safefql_model:
            print("=== Ours (SafeFQL) ===")
            trajectories.update(
                run_and_cache(args.safefql_model, "Ours", [1])
            )

        if not trajectories:
            print("No model paths provided. Use --safefql_base_model, --ifql_model, --safefql_model.")
            return

    plot_all(trajectories, args.output_name)


if __name__ == "__main__":
    main()

