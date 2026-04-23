import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12)

dense = np.random.normal(loc=[0, 0], scale=[0.45, 0.45], size=(90, 2))
sparse = np.random.normal(loc=[3.5, 2.8], scale=[1.1, 1.0], size=(35, 2))
pts = np.vstack([dense, sparse])

query_dense = np.array([0.25, -0.1])
query_sparse = np.array([4.4, 3.2])

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(pts[:, 0], pts[:, 1], alpha=0.35, s=35, label="Historical matches")

ax.scatter(*query_dense, s=120, marker="*", label="Dense-region query")
ax.scatter(*query_sparse, s=120, marker="*", label="Sparse-region query")

circle1 = plt.Circle(query_dense, 0.8, fill=False, linewidth=2)
circle2 = plt.Circle(query_sparse, 0.8, fill=False, linewidth=2)
ax.add_patch(circle1)
ax.add_patch(circle2)

ax.annotate(
    "High-confidence region",
    xy=query_dense,
    xytext=(-1.7, 1.6),
    arrowprops=dict(arrowstyle="->", lw=1.5),
    fontsize=11,
)
ax.annotate(
    "Low-confidence region",
    xy=query_sparse,
    xytext=(1.3, 5.0),
    arrowprops=dict(arrowstyle="->", lw=1.5),
    fontsize=11,
)

ax.set_title("KD-tree neighborhood intuition for confidence")
ax.set_xlabel("Feature axis 1")
ax.set_ylabel("Feature axis 2")
ax.legend()
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("poster/poster_images/kdtree_diagram.png", dpi=300)
plt.close()
