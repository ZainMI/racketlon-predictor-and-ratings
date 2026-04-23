import pandas as pd
import matplotlib.pyplot as plt

# load your data
df = pd.read_csv(
    "models/confidence/artifacts/experiment_1/experiment_1_results.csv"
)

# create bins (8 is a good number for slides)
df["bin"] = pd.cut(df["knn_confidence"], bins=8)

# compute average error per bin
grouped = df.groupby("bin", observed=False)["abs_error"].mean()

# plot
plt.figure(figsize=(6, 5))
grouped.plot(kind="bar")

plt.ylabel("Average Absolute Error")
plt.xlabel("Confidence (binned)")
plt.title("Density-Based Confidence vs Error (Racketlon)")

plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("racketlon_density_failure.png")
plt.show()
