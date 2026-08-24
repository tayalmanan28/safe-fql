"""Compare safety rates for FISOR, SafeIFQL, and SafeFQL (Ours)
on the Boat environment at different rejection-sampling budgets N.

SafeFQL always runs at N=1 (one-shot policy). FISOR and SafeIFQL
are evaluated at N = 1, 2, 4, 8, 16 to show how they depend on rejection
sampling to attain competitive safety.

Usage:
    python result_plots/plot_safety_rate_comparison.py
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# =====================================================================
# Data: safety rate (%) — percentage of episodes with zero violations
# Replace placeholder values with actual results.
# =====================================================================

N_values = [1, 2, 4, 8, 16]

# Safety rates (%) for each method at each N
# Format: {method_name: [rate_at_N1, rate_at_N2, rate_at_N4, rate_at_N8, rate_at_N16]}
safety_rates = {
    "FISOR":       [74.0, 78.0, 83.0, 96.0, 98.0],   # TODO: replace with actual values
    "SafeIFQL":  [72.0, 76.0, 79.0, 95.0, 98.0],   # TODO: replace with actual values
    "SafeFQL (Ours)": [100.0, 100.0, 100.0, 100.0, 100.0],  # always N=1
}

# Standard deviations (optional, set to 0 if unknown)
safety_stds = {
    "FISOR":       [1.2, 1.0, 1.1, 0.9, 0.5],   # TODO: replace
    "SafeIFQL":  [1.4, 1.2, 1.1, 0.8, 0.4],   # TODO: replace
    "SafeFQL (Ours)": [0.0, 0.0, 0.0, 0.0, 0.0],
}

# =====================================================================
# Colors
# =====================================================================

method_colors = {
    "FISOR":           "#850f67",
    "SafeIFQL":      "#00897B",
    "SafeFQL (Ours)":  "#ff9500",
}

# =====================================================================
# Plot
# =====================================================================

fig, ax = plt.subplots(figsize=(7, 4.5))

n_groups = len(N_values)
n_methods = len(safety_rates)
bar_width = 0.22
x = np.arange(n_groups)

for i, (method, rates) in enumerate(safety_rates.items()):
    offset = (i - (n_methods - 1) / 2) * bar_width
    stds = safety_stds.get(method, [0] * n_groups)
    bars = ax.bar(
        x + offset, rates, bar_width,
        yerr=stds, capsize=3,
        color=method_colors[method],
        edgecolor="white", linewidth=0.5,
        label=method, alpha=0.9, zorder=2,
        error_kw=dict(elinewidth=1.0, capthick=0.8, alpha=0.7),
    )

# Formatting
ax.set_xticks(x)
ax.set_xticklabels([f"N={n}" for n in N_values], fontsize=11, fontweight="bold")
ax.set_ylabel("Safety Rate (%)", fontsize=13, fontweight="bold")
ax.set_xlabel("Rejection Sampling Budget (N)", fontsize=13, fontweight="bold")
ax.set_ylim(0, 110)
ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
ax.grid(axis="y", alpha=0.2, zorder=0)
ax.set_axisbelow(True)
ax.legend(
    fontsize=11, loc="upper center", frameon=True,
    bbox_to_anchor=(0.5, -0.18), ncol=3,
)

plt.tight_layout()
plt.savefig("result_plots/safety_rate_comparison.pdf", dpi=300, bbox_inches="tight")
print("Saved to result_plots/safety_rate_comparison.pdf")
