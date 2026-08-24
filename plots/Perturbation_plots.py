"""
Grouped bar plot for perturbation experiments across three environments
(Boat, HalfCheetah, Ant).

- X-axis: perturbation levels [0%, 5%, 10%, 20%]
- For each perturbation level we plot two adjacent bars (no gap):
    * Reward % (relative to 0%): (reward / reward_at_0%) * 100
    * Cost
- The four perturbation groups are separated by a visible gap.
- There are 3 horizontal subplots (one per environment) in the same figure.
- Colors for Reward% and Cost are consistent across subplots.

Replace the numbers in the ``data`` dictionary with your real values.

Usage:
    python grouped_bar_plot_perturbations.py

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ----------------------
# Replace this with your real data
# Structure: env -> list of (reward, cost) for perturbations [0%, 5%, 10%, 20%]
# Example rewards here are placeholders; keep reward values (can be negative)
# and cost values >= 0.
data = {
    'Boat': [
        (-496.1, 0.0),  # 0%
        (-496.3, 0.0),  # 5%
        (-497.4, 0.0),  # 10%
        (-500.87, 0.4),  # 20%
    ],
    'HalfCheetah': [
        (2616.12, 0.0),  # 0%
        (2597.38, 0.0),  # 5%
        (2547.46, 0.0),  # 10%
        (2294.96, 0.1),  # 20%
    ],
    'Ant': [
        (2426.08, 0.0),   # 0%
        (2393.18, 0.0),   # 5%
        (2293.17, 0.2),   # 10%
        (1685.85, 1.4),  # 20%
    ],
}

stds = {
    'Boat': [
        (20.0, 0.0),  # 0%
        (25.0, 0.0),  # 5%
        (30.0, 0.0),  # 10%
        (50.0, 0.2),  # 20%
    ],
    'HalfCheetah': [
        (85.0, 0.0),  # 0%
        (90.0, 0.0),  # 5%
        (95.0, 0.0),  # 10%
        (110.0, 0.1),  # 20%
    ],
    'Ant': [
        (75.0, 0.0),  # 0%
        (80.0, 0.0),  # 5%
        (85.0, 0.2),  # 10%
        (200.0, 0.4),  # 20%
    ],
}

# ----------------------
# Plot settings (similar look & feel to your previous plot)
# ----------------------
SAVE_FIG = True
SAVE_PATH = "result_plots/perturbation_analysis_plot.pdf"
FIGSIZE = (15, 5)              # wide figure: 3 subplots horizontally
BAR_WIDTH = 0.82               # width of each individual bar (two bars per perturbation)
GAP_BETWEEN_GROUPS = 0.38      # horizontal spacing between perturbation groups
ANNOTATE = True

# Colors (consistent across all subplots)
col_reward = "#ff9500"   # reward percent
col_cost   = "#850f67"   # cost
col_reward_dark = "#cc7700"  # darker shade for reward error bars
col_cost_dark   = "#5a0a47"  # darker shade for cost error bars

perturbations = ["0%", "5%", "10%", "20%"]
num_groups = len(perturbations)
metrics_per_group = 2  # reward% and cost
group_width = metrics_per_group * BAR_WIDTH

# Create figure with 3 horizontal subplots (one row, three columns)
envs = list(data.keys())
fig, axes = plt.subplots(1, len(envs), figsize=FIGSIZE, sharey=False)
if len(envs) == 1:
    axes = [axes]

# Precompute x positions for groups (common for all subplots)
group_starts = np.arange(num_groups) * (group_width + GAP_BETWEEN_GROUPS)
pos_reward = group_starts
pos_cost   = group_starts + BAR_WIDTH
xtick_pos  = group_starts + (group_width / 4.0)

# Compute global max cost across all environments for consistent y-axis scale
global_max_cost = 0.0
for env in envs:
    env_data = data[env]
    env_stds = stds[env]
    costs = np.array([c for r, c in env_data], dtype=float)
    cost_stds = np.array([c for r, c in env_stds], dtype=float)
    max_cost_with_std = np.nanmax(costs + cost_stds)
    if max_cost_with_std > global_max_cost:
        global_max_cost = max_cost_with_std
global_max_cost = global_max_cost * 1.15 if global_max_cost > 0 else 1.0

for ax, env in zip(axes, envs):
    env_data = data[env]
    env_stds = stds[env]
    # convert to arrays
    rewards = np.array([r for r, c in env_data], dtype=float)
    costs   = np.array([c for r, c in env_data], dtype=float)
    reward_stds = np.array([r for r, c in env_stds], dtype=float)
    cost_stds   = np.array([c for r, c in env_stds], dtype=float)

    # Compute reward percent relative to 0% (first entry)
    r0 = rewards[0]
    # Handle divide-by-zero: if r0 == 0, set percent to nan
    if r0 == 0:
        reward_percent = np.full_like(rewards, np.nan)
        reward_percent_std = np.full_like(reward_stds, np.nan)
    else:
        reward_percent = (1+((rewards - r0) / abs(r0))) * 100.0
        # Convert reward std to percentage scale
        reward_percent_std = (reward_stds / abs(r0)) * 100.0

    # Create twin axis for cost
    ax_cost = ax.twinx()

    # Plot bars with error bars
    bars_r = ax.bar(pos_reward, reward_percent, width=BAR_WIDTH, label='Reward %', color=col_reward,
                    yerr=reward_percent_std, capsize=4, error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': col_reward_dark})
    bars_c = ax_cost.bar(pos_cost, costs, width=BAR_WIDTH, label='Cost', color=col_cost,
                         yerr=cost_stds, capsize=4, error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': col_cost_dark})

    # Labels and title
    ax.set_xlabel('Perturbation Level', fontweight='bold')
    # Only set ylabel for leftmost subplot
    if ax == axes[0]:
        ax.set_ylabel('Reward (%)', color=col_reward, fontweight='bold')
    if ax == axes[-1]:
        ax_cost.set_ylabel('Cost', color=col_cost, fontweight='bold')
    ax.set_title(env, fontweight='bold')

    # X ticks
    ax.set_xticks(xtick_pos)
    ax.set_xticklabels(perturbations)

    # Color ticks
    ax.tick_params(axis='y', labelcolor=col_reward)
    ax_cost.tick_params(axis='y', labelcolor=col_cost)

    # Y-limits: give some headroom
    # Reward %: center at 100 for 0% and allow some margin depending on data
    rp_min = np.nanmin(reward_percent)
    rp_max = np.nanmax(reward_percent)
    if np.isnan(rp_min) or np.isnan(rp_max):
        ax.set_ylim(0, 120)
    else:
        margin = max(5.0, 0.05 * (rp_max - rp_min if rp_max != rp_min else rp_max))
        low = min(0.0, rp_min - margin)
        high = rp_max + margin
        ax.set_ylim(low, high)

    # Cost y-limits (use global max for consistent scale across all plots)
    ax_cost.set_ylim(0, global_max_cost)

    # Annotate bars
    if ANNOTATE:
        # reward percent annotations
        for bar, val in zip(bars_r, reward_percent):
            h = bar.get_height()
            if np.isnan(val):
                txt = 'n/a'
            else:
                txt = f"{val:0.1f}%"
            ax.annotate(txt,
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3),
                        textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)

        # cost annotations
        for bar, val in zip(bars_c, costs):
            h = bar.get_height()
            ax_cost.annotate(f"{val:0.2f}",
                              xy=(bar.get_x() + bar.get_width() / 2, h),
                              xytext=(0, 3),
                              textcoords='offset points',
                              ha='center', va='bottom', fontsize=8)

# Create a single legend for the whole figure (using proxy artists) and place it above the subplots
proxy = [Patch(facecolor=col_reward, edgecolor='k'), Patch(facecolor=col_cost, edgecolor='k')]
fig.legend(proxy, ['Reward (%)', 'Cost'], loc='upper center', ncol=2, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.93])  # leave space on top for the legend

if SAVE_FIG:
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {SAVE_PATH}")