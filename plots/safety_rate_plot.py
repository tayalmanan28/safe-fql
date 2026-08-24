# Creating a clear, attractive bar chart of safety rates for each method and showing the underlying table.
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# Data extracted from the image/table
methods = ["BCQ-Lag", "BEAR-Lag", "CPQ", "CoptiDICE", "C2IQL", "FISOR", "Ours"]
safety_rates = [80, 71, 54, 87, 70, 85.7, 100]

# Create a DataFrame and display it to the user
df = pd.DataFrame({"Method": methods, "Safety Rate (%)": safety_rates})

# Use the helper to display the DataFrame in a friendly table view
try:
    from caas_jupyter_tools import display_dataframe_to_user
    display_dataframe_to_user("Safety rates by method", df)
except Exception:
    # If the helper isn't available, just print the DataFrame
    print(df)

# Create an attractive bar chart
cmap = cm.get_cmap("tab20")
# colors = cmap(np.linspace(0, 1, len(methods)))
colors = [
    "#2E2D2D",  # BCQ-Lag
    "#b3b3b3",  # BEAR-Lag
    "#cdb3ff",  # CPQ
    "#0091ff",  # CoptiDICE
    "#740cad",  # C2IQL
    "#850f67",  # FISOR
    "#ff9500",  # Ours
]

fig, ax = plt.subplots(figsize=(7, 4))

# no gap between bars
bars = ax.bar(methods, safety_rates, color=colors, edgecolor='black', linewidth=0.0, width=1.0)

# Styling
ax.set_title("Safety Rate (%) by Method", fontsize=16, weight='bold')
ax.set_ylabel("Safety Rate (%)", fontsize=13)
ax.set_ylim(50, 102)  # leave headroom for labels
ax.set_xlabel("")  # no extra label to keep it clean

# Light horizontal grid for readability
ax.yaxis.grid(True, linestyle='--', linewidth=0.6, alpha=0.35)
ax.set_axisbelow(True)  # grid behind the bars

# Annotate each bar with its value
# for bar, val in zip(bars, safety_rates):
#     height = bar.get_height()
#     ax.text(
#         bar.get_x() + bar.get_width() / 2,
#         height + 2,
#         f"{val:.1f}" if (val % 1) != 0 else f"{int(val)}",
#         ha='center',
#         va='bottom',
#         fontsize=11,
#         fontweight='semibold'
#     )

# Tidy layout
plt.xticks(rotation=0, fontweight='bold', fontsize=11)
plt.tight_layout()

# Save figure to disk and show
out_path = "result_plots/safety_rates_bar.pdf"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
