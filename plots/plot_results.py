import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# =====================================================================
# Data: (reward, cost) per method per environment
# =====================================================================

data = {
    "Boat": {
        "BEAR-Lag": (-569.9, 16.6),
        "CPQ": (-623, 74.09),
        "CoptiDICE": (-1056.26, 17.16),
        "C2IQL": (-780, 23.65),
        "FISOR\n(Rejection Sampling)": (-660, 3.56),
        "SafeIFQL\n(Rejection Sampling)": (-637, 1.25),
        "Ours": (-497, 0.0),
        "Ours+CP": (-499, 0.0),
    },
    "HalfCheetah": {
        "BEAR-Lag": (2720, 156),
        "CPQ": (1109, 70),
        "CoptiDICE": (1707, 0),
        "C2IQL": (2851, 252),
        "FISOR\n(Rejection Sampling)": (2166, 0),
        "SafeIFQL\n(Rejection Sampling)": (2366, 0),
        "Ours": (2593, 0),
        "Ours+CP": (2590, 0),
    },
    "Hopper": {
        "BEAR-Lag": (711, 173),
        "CPQ": (288, 100),
        "CoptiDICE": (328, 79),
        "C2IQL": (340, 32),
        "FISOR\n(Rejection Sampling)": (66.12, 1.6),
        "SafeIFQL\n(Rejection Sampling)": (156.12, 1.4),
        "Ours": (302.3, 1.02),
        "Ours+CP": (290, 0.0),
    },
    "Ant": {
        "BEAR-Lag": (-2999, 0),
        "CPQ": (-2998, 0),
        "CoptiDICE": (2968, 133),
        "C2IQL": (2965, 96),
        "FISOR\n(Rejection Sampling)": (1945, 0),
        "SafeIFQL\n(Rejection Sampling)": (2145, 0),
        "Ours": (2416, 0),
        "Ours+CP": (2416, 0),
    },
    "Walker2D": {
        "BEAR-Lag": (2678, 17),
        "CPQ": (312, 3.5),
        "CoptiDICE": (504, 38),
        "C2IQL": (1902, 31),
        "FISOR\n(Rejection Sampling)": (331.2, 1.5),
        "SafeIFQL\n(Rejection Sampling)": (342.2, 1.1),
        "Ours": (351, 0.26),
        "Ours+CP": (349, 0.0)
    },
    "Swimmer": {
        "BEAR-Lag": (37, 27),
        "CPQ": (23, 12),
        "CoptiDICE": (163, 387),
        "C2IQL": (107, 336),
        "FISOR\n(Rejection Sampling)": (-16.42, 1.0),
        "SafeIFQL\n(Rejection Sampling)": (-14.42, 0.6),
        "Ours": (-10.32, 0.04),
        "Ours+CP": (-10.34, 0.0)
    },
}

# Standard deviations: (std_reward, std_cost) per method per environment
stds = {
    "Boat": {
        "BEAR-Lag": (70.0, 6.5),
        "CPQ": (75.0, 7.0),
        "CoptiDICE": (55.0, 4.5),
        "C2IQL": (45.0, 5.0),
        "FISOR\n(Rejection Sampling)": (30.0, 3.0),
        "SafeIFQL\n(Rejection Sampling)": (30.0, 3.0),
        "Ours": (20.0, 0.01),
        "Ours+CP": (20.0, 0.0)
    },
    "HalfCheetah": {
        "BEAR-Lag": (170.0, 9.0),
        "CPQ": (140.0, 6.0),
        "CoptiDICE": (185.0, 2.0),
        "C2IQL": (90.0, 8.0),
        "FISOR\n(Rejection Sampling)": (90.0, 1.0),
        "SafeIFQL\n(Rejection Sampling)": (90.0, 1.0),
        "Ours": (85.0, 0.0),
        "Ours+CP": (85.0, 0.0)
    },
    "Hopper": {
        "BEAR-Lag": (30.0, 9.0),
        "CPQ": (20.0, 4.0),
        "CoptiDICE": (65.0, 3.5),
        "C2IQL": (22.0, 2.5),
        "FISOR\n(Rejection Sampling)": (8.0, 1.0),
        "SafeIFQL\n(Rejection Sampling)": (8.0, 1.0),
        "Ours": (6.0, 0.1),
        "Ours+CP": (6.0, 0.0)
    },
    "Ant": {
        "BEAR-Lag": (40.0, 2.0),
        "CPQ": (40.0, 2.0),
        "CoptiDICE": (85.0, 7.0),
        "C2IQL": (80.0, 6.0),
        "FISOR\n(Rejection Sampling)": (70.0, 2.0),
        "SafeIFQL\n(Rejection Sampling)": (72.0, 2.0),
        "Ours": (75.0, 0.1),
        "Ours+CP": (77.0, 0.0)
    },
    "Walker2D": {
        "BEAR-Lag": (85.0, 2.0),
        "CPQ": (60.0, 2.1),
        "CoptiDICE": (50.0, 2.5),
        "C2IQL": (80.0, 3.0),
        "FISOR\n(Rejection Sampling)": (10.0, 1.0),
        "SafeIFQL\n(Rejection Sampling)": (8.0, 1.0),
        "Ours": (8.0, 1.0),
        "Ours+CP": (8.0, 0.0)
    },
    "Swimmer": {
        "BEAR-Lag": (17.0, 10.0),
        "CPQ": (15.0, 5.0),
        "CoptiDICE": (10.0, 12.0),
        "C2IQL": (17.0, 11.0),
        "FISOR\n(Rejection Sampling)": (3.0, 0.5),
        "SafeIFQL\n(Rejection Sampling)": (1.1, 0.1),
        "Ours": (1.0, 0.05),
        "Ours+CP": (1.0, 0.0)
    },
}

# Default fallback stds if a particular entry is missing
DEFAULT_STD_COST = 1.0
DEFAULT_STD_REWARD = 2.0
DEFAULT_STD_COST = 1.0

# =====================================================================
# Methods & colors
# =====================================================================

methods = [
    "BEAR-Lag",
    "CPQ",
    "CoptiDICE",
    "C2IQL",
    "FISOR\n(Rejection Sampling)",
    "SafeIFQL\n(Rejection Sampling)",
    "Ours",
    "Ours+CP",
]

COLOR_REWARD = "#ff9500"
COLOR_COST = "#850f67"

# =====================================================================
# Plot: 3x2 grid of bar plots (3 rows, 2 columns)
# =====================================================================

env_names = list(data.keys())
fig, axes = plt.subplots(3, 2, figsize=(14, 16))
axes_flat = axes.flatten()

bar_width = 0.35  # width of each bar (reward / cost)

for ax_idx, (ax, env_name) in enumerate(zip(axes_flat, env_names)):
    env_data = data[env_name]
    env_stds = stds.get(env_name, {})

    n_methods = len(methods)
    x = np.arange(n_methods)

    rewards = []
    costs = []
    reward_errs = []
    cost_errs = []

    for m in methods:
        r, c = env_data[m]
        rewards.append(r)
        costs.append(c)
        sr, sc = env_stds.get(m, (DEFAULT_STD_REWARD, DEFAULT_STD_COST))
        reward_errs.append(sr)
        cost_errs.append(sc)

    # Compute reward y-axis floor so bars are grounded to the bottom
    min_reward = min(rewards)
    max_reward = max(rewards)
    reward_range = max(1.0, max_reward - min_reward)
    y_floor = min_reward - 0.08 * reward_range  # small margin below lowest bar
    bar_heights = [r - y_floor for r in rewards]

    # Reward bars (left of center) — grounded to y_floor
    bars_r = ax.bar(
        x - bar_width / 2, bar_heights, bar_width,
        bottom=y_floor,
        yerr=reward_errs, capsize=3,
        color=COLOR_REWARD, edgecolor="white", linewidth=0.5,
        label="Reward", alpha=0.9, zorder=2,
        error_kw=dict(elinewidth=1.0, capthick=0.8, alpha=0.7),
    )
    ax.set_ylim(bottom=y_floor, top=max_reward + 0.12 * reward_range)

    # Cost bars (right of center) — plotted on a twin axis
    ax2 = ax.twinx()
    bars_c = ax2.bar(
        x + bar_width / 2, costs, bar_width,
        yerr=cost_errs, capsize=3,
        color=COLOR_COST, edgecolor="white", linewidth=0.5,
        label="Cost", alpha=0.9, zorder=2,
        error_kw=dict(elinewidth=1.0, capthick=0.8, alpha=0.7),
    )

    # X-axis labels — use short display names to avoid overlap
    short_names = [
        "BEAR-Lag", "CPQ", "CoptiDICE", "C2IQL",
        "FISOR\n(R.S.)", "SafeIFQL\n(R.S.)", "Ours", "Ours+CP",
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=13, fontweight="bold", ha="center", rotation=90)

    # Title
    ax.set_title(env_name, fontweight="bold", fontsize=18)

    # Y-axis labels (only on edges of 2-column layout)
    if ax_idx % 2 == 0:
        ax.set_ylabel("Reward", fontsize=15, fontweight="bold", color=COLOR_REWARD)
    if ax_idx % 2 == 1:
        ax2.set_ylabel("Cost", fontsize=15, fontweight="bold", color=COLOR_COST)

    # Style the y-axis tick colors
    ax.tick_params(axis="y", labelcolor=COLOR_REWARD, labelsize=12)
    ax2.tick_params(axis="y", labelcolor=COLOR_COST, labelsize=12)

    # Force cost axis to start at 0 (prevents purple bars from floating)
    max_cost = max(costs) + max(cost_errs)
    ax2.set_ylim(bottom=0, top=max_cost * 1.15)

    # Limit to ~5 round-number y-ticks on both axes
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))

    # Grid on reward axis only (sparse, matching 5 ticks)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.set_axisbelow(True)

# Shared legend at the bottom
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=COLOR_REWARD, edgecolor="white", label="Reward"),
    Patch(facecolor=COLOR_COST, edgecolor="white", label="Cost"),
]
fig.legend(
    handles=legend_elements, loc="lower center",
    ncol=2, fontsize=16, frameon=True,
    bbox_to_anchor=(0.5, -0.03),
)

plt.subplots_adjust(
    hspace=0.50, wspace=0.20,
    top=0.97, bottom=0.07, left=0.07, right=0.93,
)
plt.savefig("result_plots/evaluation_results.pdf", dpi=300, bbox_inches="tight")
print("Saved to result_plots/evaluation_results.pdf")
