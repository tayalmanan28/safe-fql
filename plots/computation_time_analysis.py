from turtle import color
import numpy as np
import matplotlib.pyplot as plt


data = {
    1:  (-506.55, 26.51, 1.74),
    2:  (-498.63, 13.6, 1.75),
    4:  (-496.96, 6.09, 1.78),
    8:  (-496.10, 0.0, 1.81),
    16: (-498.23, 0.0, 1.82),
    32: (-507.20, 0.0, 1.89),
    64: (-512.46, 0.0, 1.95),
    128:(-519.53, 0.0, 2.19),
}

stds = {
    1:  (15.0, 6.0, 0.02),
    2:  (18.0, 4.0, 0.02),
    4:  (24.0, 1.0, 0.02),
    8:  (15.0, 0.0, 0.02),
    16: (17.0, 0.0, 0.02),
    32: (21.0, 0.0, 0.03),
    64: (27.0, 0.0, 0.03),
    128:(22.0, 0.0, 0.05),
}

# ----------------------
# Plot settings
# ----------------------
SAVE_FIG = True
SAVE_PATH = "result_plots/computation_analysis_plot.pdf"
FIGSIZE = (12, 6)
BAR_WIDTH = 2.2             # width of each individual bar (three bars will be adjacent)
GAP_BETWEEN_GROUPS = 0.82    # horizontal spacing between triplets
ANNOTATE = True              # annotate bars with their numeric values

# ----------------------
# Prepare arrays
# ----------------------
Ns = sorted(list(data.keys()))
rewards = np.array([data[n][0] for n in Ns], dtype=float)  # negative numbers expected
costs   = np.array([data[n][1] for n in Ns], dtype=float)
times   = np.array([data[n][2] for n in Ns], dtype=float)

# Extract standard deviations
reward_stds = np.array([stds[n][0] for n in Ns], dtype=float)
cost_stds   = np.array([stds[n][1] for n in Ns], dtype=float)
time_stds   = np.array([stds[n][2] for n in Ns], dtype=float)

# We will plot rewards transformed so -520 is at origin and -480 at top
# Transform: bar_height = reward - (-520) = reward + 520
reward_min_val = -560
transformed_rewards = rewards - reward_min_val  # e.g., -500 becomes 20
# Note: reward_stds don't need transformation since they are relative differences

n_groups = len(Ns)
metrics_per_group = 3
group_width = metrics_per_group * BAR_WIDTH

# x positions for group starts
group_starts = np.arange(n_groups) * (group_width + GAP_BETWEEN_GROUPS)

# positions for each bar inside a group (no gap between them)
pos_reward = group_starts
pos_cost   = group_starts + BAR_WIDTH
pos_time   = group_starts + 2 * BAR_WIDTH

# xticks are at center of the triplet
xtick_pos = group_starts + (group_width / 3.0)

# ----------------------
# Create figure and axes
# ----------------------
fig, ax_reward = plt.subplots(figsize=FIGSIZE)

# Two additional right-side y-axes
ax_cost = ax_reward.twinx()            # right-side inner
ax_time = ax_reward.twinx()            # right-side outer
# move the third axis (ax_time) further to the right
ax_time.spines["right"].set_position(("axes", 1.08))
# make sure the spines are visible
ax_time.spines["right"].set_visible(True)

# Avoid covering the plotting area for ax_time
ax_cost.set_frame_on(True)
ax_time.set_frame_on(True)

# Colors for visual separation
col_reward = "#ff9500"   # orange
col_reward_dark = "#cc7700"  # darker shade for reward error bars
col_cost   = "#850f67"   # maroon
col_cost_dark = "#5a0a47"   # darker shade for cost error bars
col_time   = "#740cad"   # purple
col_time_dark = "#590992"   # darker shade for time error bars

# Plot bars on respective axes with error bars
bars_r = ax_reward.bar(pos_reward, transformed_rewards, width=BAR_WIDTH, label='Reward (neg)', color=col_reward,
                       yerr=reward_stds, capsize=4, error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': col_reward_dark})
bars_c = ax_cost.bar(pos_cost, costs, width=BAR_WIDTH, label='Cost', color=col_cost,
                     yerr=cost_stds, capsize=4, error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': col_cost_dark})
bars_t = ax_time.bar(pos_time, times, width=BAR_WIDTH, label='Computation Time (s)', color=col_time,
                     yerr=time_stds, capsize=4, error_kw={'elinewidth': 1.5, 'capthick': 1.5, 'ecolor': col_time_dark})

# Axis labels and title
ax_reward.set_xlabel('Number of Samples (N)', fontweight='bold')
ax_reward.set_ylabel('Reward', color=col_reward, fontweight='bold')
ax_cost.set_ylabel('Cost', color=col_cost, fontweight='bold')
ax_time.set_ylabel('Computation Time (s)', color=col_time, fontweight='bold')
# plt.title('Results vs N: Reward (negative), Cost, Computation Time')

# Set xticks
ax_reward.set_xticks(xtick_pos)
ax_reward.set_xticklabels([str(n) for n in Ns])

# Color ticks/labels to match bars
# create ticks to bold
ax_reward.tick_params(axis='y', labelcolor=col_reward, labelsize=10)
ax_cost.tick_params(axis='y', labelcolor=col_cost, labelsize=10)
ax_time.tick_params(axis='y', labelcolor=col_time, labelsize=10)

# Set sensible y-limits (give a small top margin)
# For reward axis: range -520 to -480, with -520 at bottom (origin) and -480 at top
# We plot (reward - (-520)) = (reward + 520) so that -520 maps to 0 and -480 maps to 40
reward_min, reward_max = -560, -480
ax_reward.set_ylim(0, reward_max - reward_min)  # 0 to 40
if costs.size:
    ax_cost.set_ylim(0, np.nanmax(costs) * 1.5 if np.nanmax(costs) > 0 else 1.0)
if times.size:
    ax_time.set_ylim(0, np.nanmax(times) * 1.15 if np.nanmax(times) > 0 else 1.0)

# Replace left-axis tick labels to show negative values (-520 at bottom, -480 at top)
yticks_reward = ax_reward.get_yticks()
# Map tick values back to actual reward: tick 0 -> -520, tick 40 -> -480
ytick_labels_reward = [f"{reward_min + y:.0f}" for y in yticks_reward]
ax_reward.set_yticklabels(ytick_labels_reward)

# Create a combined legend using proxy artists
from matplotlib.patches import Patch
proxy_artists = [Patch(facecolor=col_reward, edgecolor='k'), Patch(facecolor=col_cost, edgecolor='k'), Patch(facecolor=col_time, edgecolor='k')]
# ax_reward.legend(proxy_artists, ['Reward (neg)', 'Cost', 'Computation Time (s)'], loc='upper left')

# Annotate bars with values on their respective axes
if ANNOTATE:
    def annotate(ax, bar_container, vals, fmt='{:.2f}', yoffset=3):
        for bar, v in zip(bar_container, vals):
            height = bar.get_height()
            ax.annotate(fmt.format(v),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, yoffset),
                        textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)

    # For rewards, show the original negative numbers in the annotation
    annotate(ax_reward, bars_r, rewards, fmt='{:0.1f}')
    annotate(ax_cost,   bars_c, costs,   fmt='{:0.1f}')
    annotate(ax_time,   bars_t, times,   fmt='{:0.2f}')

# Grid only for reward axis (left)
ax_reward.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)

# Add dashed vertical lines separating bars for different values of N
for i in range(1, n_groups):
    # Position the line between the end of previous group and start of current group
    # x_line = (group_starts[i-1] + group_width + group_starts[i]) / 2
    x_line = group_starts[i-1] + 5*group_width/6 + (GAP_BETWEEN_GROUPS / 2)
    ax_reward.axvline(x=x_line, color='gray', linestyle='--', linewidth=1, alpha=0.7)

plt.tight_layout()

if SAVE_FIG:
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {SAVE_PATH}")
