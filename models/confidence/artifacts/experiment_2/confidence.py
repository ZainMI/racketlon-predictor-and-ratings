import pandas as pd
import matplotlib.pyplot as plt

# Load your experiment 2 data
df = pd.read_csv(
    "models/confidence/artifacts/experiment_2/experiment_2_results.csv"
)

# Create equal-sized bins (quantiles)
df["bin"] = pd.qcut(df["consistency_confidence"], q=8)

# Compute average error per bin
grouped = df.groupby("bin", observed=False)["abs_error"].mean()

# --- Fix labels: round to exactly 3 decimals ---
grouped.index = [
    f"{round(interval.left, 3):.3f}–{round(interval.right, 3):.3f}"
    for interval in grouped.index
]

# --- Plot ---
plt.figure(figsize=(6, 5))

grouped.plot(kind="bar", edgecolor="black")

plt.ylabel("Average Absolute Error", fontsize=12)
plt.xlabel("Consistency (binned)", fontsize=12)
plt.title("Consistency-Based Confidence vs Error", fontsize=14)

plt.xticks(rotation=30, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("consistency_plot.png", dpi=200)
plt.show()
