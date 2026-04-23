import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))
ax.axis("off")


def box(text, xy):
    ax.text(
        xy[0],
        xy[1],
        text,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.4", fc="#e6f0ff", ec="#1a4c8c", lw=2),
        fontsize=12,
    )


# Boxes
box("Scraper\n(match_scraper.py)", (0.1, 0.5))
box("Clean Data\n(clean_matches.py)", (0.3, 0.5))
box("Feature Builder\n(features.py)", (0.5, 0.5))
box("Train Model\n(GBDT)", (0.7, 0.5))
box("Inference API\n(funcs.py)", (0.9, 0.5))

# Arrows
arrowprops = dict(arrowstyle="->", lw=2)

ax.annotate("", xy=(0.23, 0.5), xytext=(0.17, 0.5), arrowprops=arrowprops)
ax.annotate("", xy=(0.43, 0.5), xytext=(0.37, 0.5), arrowprops=arrowprops)
ax.annotate("", xy=(0.63, 0.5), xytext=(0.57, 0.5), arrowprops=arrowprops)
ax.annotate("", xy=(0.83, 0.5), xytext=(0.77, 0.5), arrowprops=arrowprops)

plt.savefig("poster/poster_images/pipeline.png", bbox_inches="tight", dpi=300)
plt.close()
