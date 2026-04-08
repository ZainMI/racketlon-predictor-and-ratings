# 🏓 Racketlon Match Predictor — Implementation Overview

This project implements a full **end-to-end ML pipeline** for predicting racketlon match outcomes and scorelines across four sports:

- Table Tennis (TT)
- Badminton (BD)
- Squash (SQ)
- Tennis (TN)

The system is designed to:

- operate on **chronological match data**
- avoid **data leakage**
- support **unseen player pairings**
- provide **interpretable and structured predictions**

---

# 🏗️ System Architecture

The project is composed of three main layers:

## 1. Data & Feature Pipeline

```
matches_cleaned.csv → features_v4.py → data/data.csv
```

## 2. Model Layer

- Baseline: Ridge regression
- Tier 1: CatBoost (tabular)
- Tier 2: Player embedding neural network (PyTorch)

## 3. Inference Layer

- Synthetic matchup builder
- Score reconstruction logic
- Pretty-print / API-style prediction output

---

# 📊 Data Pipeline

## Input Schema

Raw match data (`matches_cleaned.csv`) contains:

- players (`team1_players`, `team2_players`)
- scores per sport (`TT_p1`, `TT_p2`, etc.)
- match timestamps

---

## Chronological Ordering

All processing is done in **strict time order**:

```python
df = df.sort_values("datetime")
```

This guarantees:

- no future information is used
- all features are pre-match

---

## Feature Construction

Implemented in:

```python
features_v4.py
```

Each row represents a **single match**, with features computed **before updating state**.

---

## Feature Categories

### 1. Current Ratings (Margin-Elo)

Stateful structure:

```python
self.R[player][sport]
self.games[player][sport]
```

Update rule:

```python
error = actual_diff - predicted_diff
step = eta * error * gradient
```

Stored features:

```text
TT_rating_p1
TT_rating_p2
TT_rating_diff
```

---

### 2. Snapshot Ratings (Monthly)

Computed using:

- batched monthly updates
- iterative smoothing
- alpha fitting (in original system)

Stored as:

```text
TT_snapshot_rating_p1
TT_snapshot_rating_p2
TT_snapshot_rating_diff
```

Key property:

- **constant within a month**
- independent of match order inside that month

---

### 3. Head-to-Head (H2H)

Maintained per player pair:

```python
(pair_key) → H2HStats
```

Tracked:

- games played
- winrate
- average diff
- last match date

Stored as:

```text
TT_h2h_games
TT_h2h_winrate_p1
TT_h2h_avg_diff_p1
TT_h2h_days_since_last
```

---

### 4. Experience

Tracks number of matches per player:

```text
TT_games_p1
TT_games_p2
TT_games_diff
```

Used as a proxy for:

- confidence
- rating reliability

---

## Targets

Per sport:

```text
TT_y_diff
TT_y_total
```

Match-level:

```text
y_total_diff
y_winner_p1
```

---

# 🧠 Model Implementations

---

## 🥉 Baseline (Ridge)

File:

```python
baseline.py
```

### Structure

For each sport:

```python
model_diff = Ridge()
model_total = Ridge()
```

Features:

```text
delta_diff
delta_total
sum_total
```

---

## 🥈 Tier 1 (CatBoost)

File:

```python
catboost_regressor.py
```

### Design

- Separate model per sport
- Two regressors per sport:
  - diff
  - total

### Feature Filtering

Explicitly removes:

```python
if "_pred_diff" in c: continue
if c.startswith("has_"): continue
```

Ensures:

- no redundant transforms
- no leakage

---

### Training Loop

```python
for sport in SPORTS:
    model_diff.fit(X_train, y_diff)
    model_total.fit(X_train, y_total)
```

Predictions are clipped:

```python
pred_diff = np.clip(..., -21, 21)
pred_total = np.clip(..., 0, 42)
```

---

## 🥇 Tier 2 (Player Embedding Model)

File:

```python
player_embedding_fin.py
```

---

## Architecture

```text
Embedding(p1) → e1
Embedding(p2) → e2

Feature vector:
[e1, e2, e1 - e2, e1 * e2, numeric_features]

→ MLP → (diff, total)
```

---

### Model Definition

```python
self.embed = nn.Embedding(n_players + 1, embed_dim)
self.backbone = nn.Sequential(...)
self.head = nn.Linear(...)
```

---

### Input Construction

```python
x = torch.cat([
    e1,
    e2,
    e1 - e2,
    e1 * e2,
    numeric_features
], dim=1)
```

---

### Loss

```python
loss = SmoothL1(diff) + SmoothL1(total)
```

---

### Training

- batched gradient descent
- AdamW optimizer
- early stopping on validation loss

---

# 🔁 Inference Pipeline

---

## Problem

We must predict for **unseen matchups**.

We cannot rely on:

```text
latest row where p1 vs p2 occurred
```

---

## Solution: Synthetic Row Construction

Implemented in:

```python
build_synthetic_match_row(...)
```

---

### Steps

#### 1. Get player states

```python
s1 = get_latest_player_state(df, player1)
s2 = get_latest_player_state(df, player2)
```

Each player is normalized to be `p1`.

---

#### 2. Merge features

```python
rating_diff = r1 - r2
games_diff = g1 - g2
```

---

#### 3. Add H2H

If exists:

```python
pair_row = get_latest_pair_h2h_row(...)
```

Else:

```python
h2h_games = 0
h2h_winrate = 0.5
```

---

#### 4. Construct final row

```python
return pd.Series({
    col: row.get(col, np.nan)
})
```

---

# 🎯 Score Reconstruction

Models output:

```text
pred_diff
pred_total
```

Converted to scores:

```python
s1 = 0.5 * (total + diff)
s2 = 0.5 * (total - diff)
```

---

## Non-Tennis Sports

Forced into valid format:

```text
21 - x  or  x - 21
```

---

## Tennis

Two modes:

### Independent

```python
decode_full_game_score(...)
```

### Racketlon Stop Rule

```python
decode_tennis_score(...)
```

Uses:

```text
running_diff_before_tn
```

---

# 💾 Model Packaging

Each model saves:

```text
player_embedding_package.pt
data_snapshot.pkl
metrics.json
plots/
```

---

## Package Contents

```python
{
  "state_dicts": model weights,
  "feature_cols": feature list,
  "player_to_idx": mapping,
  "model_meta": normalization stats
}
```

---

## Loading

```python
predictor = PredictorPackage.load(path)
```

---

## Prediction

```python
predictor.predict_pair(player1, player2)
```

---

# 🧩 Key Design Decisions

---

## 1. Chronological training

- avoids leakage
- simulates real prediction

---

## 2. Explicit feature filtering

- removes derived features
- reduces redundancy

---

## 3. Dual rating system

- snapshot → long-term
- current → short-term

---

## 4. Synthetic inference

- enables unseen pair prediction
- decouples model from dataset structure

---

## 5. Separate sport models

- each sport has different dynamics
- avoids forcing shared distributions

---

# 🏁 Summary

This system implements:

- a leakage-safe feature pipeline
- multiple model tiers
- a reusable prediction package
- a generalizable inference system

The implementation prioritizes:

- correctness of temporal logic
- flexibility of feature construction
- extensibility for future models

---

If you want next, I can:

- convert this into a **paper-style Methods section**
- or add **diagrams (model + pipeline)** for your project submission
