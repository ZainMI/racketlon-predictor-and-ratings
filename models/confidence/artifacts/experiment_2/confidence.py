import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    "models/confidence/artifacts/experiment_2/experiment_2_results.csv"
)

# Use fewer bins for clarity
df["bin"] = pd.qcut(df["consistency_confidence"], q=6)

grouped = df.groupby("bin", observed=False)["abs_error"].mean()

# Clean labels
grouped.index = [f"Bin {i+1}" for i in range(len(grouped))]

plt.figure(figsize=(6, 5))

grouped.plot(kind="bar", edgecolor="black")

plt.ylabel("Average Absolute Error", fontsize=12)
plt.xlabel("Consistency (low → high)", fontsize=12)
plt.title("Consistency-Based Confidence vs Error", fontsize=14)

plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("consistency_plot.png", dpi=200)
plt.show()
