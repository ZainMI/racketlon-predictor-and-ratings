# 🏓 Racketlon Match Predictor

This project implements a full **end-to-end machine learning pipeline** for predicting racketlon match outcomes.

Racketlon is a multi-sport competition where two players compete sequentially across four sports:

- Table Tennis (TT)
- Badminton (BD)
- Squash (SQ)
- Tennis (TN)

The winner is determined by the total number of points accumulated across all four sports. This repository provides the tools to scrape historical data, track player ratings, train predictive models, and forecast future matchups.

---

# ⚙️ How It Works: The End-to-End Pipeline

The system is designed to operate on strict **chronological match data** to prevent data leakage. Here is how data flows from the web into a final prediction:

### 1. Data Collection (Scraping)

Historical tournament IDs are loaded from `tournament_ids.csv`. The scraper (`tournament_id_scraper.py` and `match_scraper.py`) visits Tournament Software / FIR websites, handling both legacy and modern layouts. It extracts match dates, player names, and sport-by-sport scores, outputting everything to a unified raw `matches.csv`.

### 2. Cleaning & Normalization

Raw scraped data often contains walkovers, doubles matches, or corrupted scores. The cleaning step (`data_clean.py`) filters out non-standard matches and normalizes player names (e.g., removing seeding tags like `[3]`). Consistent identity is critical because the feature pipeline relies on it to build accurate player histories. The result is `matches_cleaned.csv`.

### 3. Dynamic Ratings & Feature Engineering

Implemented in `features.py`. The system processes matches **in strict time order**. Before a match is processed, features are extracted for the two players based _only_ on their history up to that exact moment. Features include:

- **Dynamic Elo-style Ratings**: A custom, sport-specific rating system that translates rating differences into expected score differentials via a bounded nonlinear mapping (`tanh`).
- **Recent Form**: Rolling windows tracking recent score margins, residual performance (actual vs. expected), and momentum.
- **Head-to-Head (H2H)**: Pairwise history, both overall and sport-specific.

After features are generated, the match results update the underlying player states.

### 4. Model Training

The final production model uses **CatBoost** gradient-boosted decision trees. For each sport, three separate regressors are trained:

1. **Base Differential Model**: Predicts the point difference using stable, long-term features (ratings, overall H2H).
2. **Residual Correction Model**: Refines the base prediction using short-term form and momentum.
3. **Total Points Model**: Predicts the total points scored in the sport, which helps determine the pace and realistic scoreline.

### 5. Inference & Score Reconstruction

To predict a match between two players (even if they have never played each other), the inference system:

1. Looks up the latest saved state for both players from `inference_state.pkl`.
2. Synthesizes a new feature row representing the hypothetical matchup.
3. Feeds the row through the sport-specific models.
4. **Reconstructs realistic scores**: Converts the continuous predictions (e.g., diff of `+3.5`, total of `38.5`) into valid racketlon scorelines (e.g., `21-17`).

---

# 💻 Usage & API

The entire pipeline is wrapped in `funcs.py`, which provides a clean service-layer interface for application integration.

**Running a Prediction:**

```python
from funcs import matchup_bundle

# Predicts outcome, plus detailed state metrics for both players
result = matchup_bundle("zain magdon-ismail", "patrick moran")

print(result["prediction"]["winner"])
```

**Rebuilding the Pipeline:**
You can trigger a full end-to-end rebuild (optionally scraping new data, cleaning, generating features, and training the CatBoost models):

```python
from funcs import rebuild_all

# Complete refresh of data, features, and model weights
rebuild_all(scrape=False)
```

---

## 🧠 Model Tiers Explored

While **CatBoost** is the current production model, the project explored several approaches to evaluate how much performance comes from the model versus the features:

- **Baseline (Ridge Regression)**: A regularized linear model. Proved that the engineered features (like rating diffs and recent form) already capture the vast majority of the predictive signal.
- **Tier 2 (Player Embedding Neural Network)**: A PyTorch-based MLP that learned 16-dimensional embeddings for each player alongside the tabular features. It performed similarly to Ridge/CatBoost, reinforcing that the hand-engineered representation is extremely strong.

---

## 🧩 Key Design Decisions

- **Chronological Strictness**: Avoiding "future leakage" is the hardest part of sports modeling. Sorting data by datetime and generating features strictly from past matches ensures integrity for future deployment.
- **Regression over Classification**: We predict sport differentials and totals instead of just "who wins." This preserves valuable margin-of-victory information and allows us to reconstruct realistic game scores.
- **Synthetic Inference**: Decoupling the training matrix from the inference state allows the model to instantly simulate hypothetical matchups without retraining.
- **Separate Sport Models**: Because Racketlon combines four distinct sports, each sport has its own unique scoring pacing and distribution, requiring independently tuned regressors.
