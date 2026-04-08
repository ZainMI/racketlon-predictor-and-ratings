Absolutely — here’s a **low-level, implementation-oriented project document**.
This is written more like an internal engineering spec than a high-level README.

---

# Racketlon Predictor — Implementation Notes

## Repository purpose

This repository implements a predictive system for racketlon matches using:

- chronological feature construction from raw match history
- per-sport regression models for:
  - score difference
  - score total

- match-level score reconstruction
- synthetic feature construction for unseen player pairings

The project currently has two main model families:

- **Tier 1:** CatBoost on engineered tabular features
- **Tier 2:** PyTorch player-embedding MLP on engineered tabular features

---

# 1. Core data flow

## 1.1 Raw input

Primary raw source:

```text
data/matches_cleaned.csv
```

Expected columns include:

```text
mode,tournament_id,match_date,match_time,draw,draw_id,round,duration,location,
team1_players,team2_players,team1_player_ids,team2_player_ids,
winner_side,status_message,
TT_p1,TT_p2,BD_p1,BD_p2,SQ_p1,SQ_p2,TN_p1,TN_p2,...
```

Important raw fields used by the implementation:

- `match_date`
- `match_time`
- `team1_players`, `team2_players`
- `team1_player_ids`, `team2_player_ids`
- `TT_p1`, `TT_p2`
- `BD_p1`, `BD_p2`
- `SQ_p1`, `SQ_p2`
- `TN_p1`, `TN_p2`

---

## 1.2 Processed feature table

Generated output:

```text
data/data.csv
```

This is the main training table for Tier 1 and Tier 2.

Each row corresponds to one historical match.

The row contains:

- identifiers
- pre-match engineered features
- post-match targets

The invariant is:

> all feature columns must be computable using only information available before the match starts

---

# 2. Feature generation implementation

Implemented in:

```text
features_v4.py
```

The file builds `data/data.csv` directly from `matches_cleaned.csv`.

---

## 2.1 Row ordering

The entire pipeline depends on strict chronological ordering.

Implementation pattern:

```python
df["datetime"] = safe_datetime(df)
df = df.sort_values("datetime").reset_index(drop=True)
```

This ordering is used for:

- current rating updates
- H2H updates
- snapshot assignment
- leakage prevention

---

## 2.2 Identity handling

Players are keyed using:

```python
player_key(row, side)
```

Behavior:

- if `USE_IDS = True` and `team{side}_player_ids` exists, use that
- otherwise fall back to normalized player name

This lets the system distinguish between:

- **display identity**: `p1_name`, `p2_name`
- **model identity**: `p1_key`, `p2_key`

This separation matters because:

- embeddings use keys
- snapshot logic uses names
- printed output uses names

---

## 2.3 Current running ratings

Current ratings are maintained by `MarginElo`.

### State structure

```python
self.R[player][sport]
self.games[player][sport]
```

Where:

- `R` stores current rating per player per sport
- `games` stores total completed matches per player per sport

### Prediction function

For each sport:

```python
x = self.R[p1][sport] - self.R[p2][sport]
pred = rating_to_score_diff(x, ALPHAS[sport])
```

### Update function

After each completed match:

```python
err = actual_diff - pred
step = eta * err * grad
step -= eta * L2 * x
```

Then:

```python
R[p1] += step
R[p2] -= step
```

The update is symmetric.

### Stored columns

For each sport, feature generation writes:

```text
TT_rating_p1
TT_rating_p2
TT_rating_diff
TT_games_p1
TT_games_p2
TT_games_diff
```

Same for `BD`, `SQ`, `TN`.

These are the columns later used for synthetic inference.

---

## 2.4 Snapshot ratings

Snapshot ratings are monthly, slower-moving skill estimates.

They are computed in memory from the raw match table rather than loaded from a separate JSON file.

### Build path

Main function:

```python
build_ratings_by_month_from_matches(df)
```

This groups matches by `(year, month)` and computes a snapshot object for each month.

Each snapshot entry contains:

```python
{
    "ratings": {
        player_name: {
            "tt": ...,
            "bd": ...,
            "sq": ...,
            "tn": ...
        }
    },
    "alphas": {...}
}
```

### Snapshot selection for a match

For each row in `data.csv`:

```python
snapshot_key, snapshot = get_last_month_snapshot(...)
```

Then the feature generator writes:

```text
TT_snapshot_rating_p1
TT_snapshot_rating_p2
TT_snapshot_rating_diff
TT_snapshot_p1_found
TT_snapshot_p2_found
```

These are raw monthly ratings, not current running ratings.

### Important distinction

- `*_rating_*` = dynamic match-by-match state
- `*_snapshot_rating_*` = month-frozen state

---

## 2.5 Head-to-head features

H2H is stored using `H2HStats`.

### State

```python
games
wins_a
sum_diff_a
last_dt
```

Pair key is canonicalized via:

```python
pair_key(a, b)
```

This ensures `(p1, p2)` and `(p2, p1)` hit the same state object.

### Per-row output

Overall:

```text
h2h_games
h2h_avg_diff_p1
h2h_winrate_p1
h2h_days_since_last
```

Per sport:

```text
TT_h2h_games
TT_h2h_avg_diff_p1
TT_h2h_winrate_p1
TT_h2h_days_since_last
```

etc.

The `_p1` suffix means:

> values are oriented from the perspective of the current row’s p1 player

---

## 2.6 Targets

For each sport:

```text
TT_y_diff
TT_y_total
```

And match-level:

```text
y_total_diff
y_winner_p1
```

Targets are written into the same row only after pre-match features have been assembled.

---

# 3. Tier 1 model implementation

Implemented in a CatBoost training file such as:

```text
catboost_regressor.py
```

or the cleaned production-style variant.

---

## 3.1 Model decomposition

There is no single joint model across all sports.

Instead:

For each sport:

- one regressor predicts `*_y_diff`
- one regressor predicts `*_y_total`

So total trained models:

```text
4 sports × 2 targets = 8 regressors
```

This decomposition keeps each target distribution simpler.

---

## 3.2 Feature filtering

Feature selection explicitly removes:

- identifiers
- target columns
- `has_*`
- old transformed columns like `*_pred_diff`
- `snapshot_total_pred_diff`
- `snapshot_winner_p1`

This is important because:

- some columns are redundant
- some columns are post-match indicators
- some columns came from older handcrafted transforms not needed by CatBoost

Implementation pattern:

```python
if c.endswith("_y_diff"): continue
if c.endswith("_y_total"): continue
if c.startswith("has_"): continue
if "_pred_diff" in c: continue
```

---

## 3.3 Training split

Training uses chronological index-based partitioning.

Typical logic:

```python
split = int(len(df) * TRAIN_RATIO)
train rows: index < split
test rows: index >= split
```

No random shuffling is used.

---

## 3.4 Prediction outputs

The CatBoost models predict:

- sport diff
- sport total

These raw outputs are then converted into legal scorelines.

---

# 4. Tier 2 model implementation

Implemented in:

```text
player_embedding_fin.py
```

---

## 4.1 Motivation

Tier 2 augments tabular features with learned player embeddings.

Instead of only using engineered comparisons, it also learns a latent vector for each player.

---

## 4.2 Player indexing

Player keys are mapped to integers using:

```python
build_player_index(df)
```

Mapping format:

```python
{
    player_key: integer_id
}
```

Index `0` is reserved for unknown players.

This mapping is serialized into the saved model package.

---

## 4.3 Architecture

Main module:

```python
class SportEmbeddingNet(nn.Module)
```

### Inputs

- `p1_idx`
- `p2_idx`
- numeric feature vector

### Embedding lookup

```python
e1 = self.embed(p1_idx)
e2 = self.embed(p2_idx)
```

### Combined tensor

The final input to the MLP is:

```python
torch.cat([e1, e2, e1 - e2, e1 * e2, x_num], dim=1)
```

This explicitly gives the model:

- each player vector
- embedding difference
- embedding interaction
- engineered numeric features

### Output head

The network outputs 2 values:

```text
[ predicted_diff , predicted_total ]
```

One model instance is trained per sport.

---

## 4.4 Loss

Tier 2 uses:

```python
nn.SmoothL1Loss()
```

Training objective:

```python
loss = loss_diff + loss_total
```

So each sport model is still dual-output, but trained jointly for that sport.

---

## 4.5 Validation / early stopping

Tier 2 uses three chronological regions:

- train
- validation
- test

Validation loss is tracked after each epoch.

Best model state is stored in memory:

```python
best_state = {k: v.detach().cpu().clone() ...}
```

Then restored at the end.

Early stopping uses:

```python
EARLY_STOPPING_PATIENCE
```

---

## 4.6 Normalization

Numeric features are standardized using only the training split:

```python
feat_mean = df_train[feature_cols].mean()
feat_std = df_train[feature_cols].std().replace(0, 1.0)
```

These are saved per sport and reused at inference time.

---

# 5. Synthetic inference implementation

This is one of the most important engineering parts of the system.

The predictor is not allowed to depend on:

> “the latest direct historical row for the same matchup”

Instead, it must support unseen pairings.

---

## 5.1 Core strategy

Build a fresh synthetic row from:

- latest state of player 1
- latest state of player 2
- latest pair H2H summary if available
- neutral H2H defaults otherwise

---

## 5.2 Player state extraction

```python
get_latest_player_state(df, player_name)
```

This finds the latest row where the player appears, then reorients that row so the player becomes logical `p1`.

This matters because `data.csv` stores many fields relative to `p1`.

---

## 5.3 Row orientation

Function:

```python
orient_row_to_player_as_p1(row, player_name)
```

This swaps:

- player names / keys
- `*_rating_p1` vs `*_rating_p2`
- `*_games_p1` vs `*_games_p2`
- snapshot p1/p2 fields

and negates directional values:

- `*_rating_diff`
- `*_games_diff`
- `*_h2h_avg_diff_p1`
- `*_snapshot_rating_diff`

and flips winrates:

- `h2h_winrate_p1`
- `*_h2h_winrate_p1`

This is necessary so that later synthetic construction can safely read “player-local” state from `..._p1`.

---

## 5.4 Pair H2H lookup

Function:

```python
get_latest_pair_h2h_row(df, player1, player2)
```

If no direct historical matchup exists, returns `None`.

If it exists, the row is also reoriented so `player1` is logical `p1`.

---

## 5.5 Synthetic row assembly

Function:

```python
build_synthetic_match_row(df, player1, player2, feature_cols)
```

This writes a dictionary containing:

### from player 1 state

- `*_rating_p1`
- `*_games_p1`
- `*_snapshot_rating_p1`
- `*_snapshot_p1_found`

### from player 2 state

- `*_rating_p2`
- `*_games_p2`
- `*_snapshot_rating_p2`
- `*_snapshot_p2_found`

### derived pairwise fields

- `*_rating_diff = rating_p1 - rating_p2`
- `*_games_diff = games_p1 - games_p2`
- `*_snapshot_rating_diff = snapshot_rating_p1 - snapshot_rating_p2`

### pair H2H defaults

if no direct pair history exists:

```python
h2h_games = 0
h2h_avg_diff_p1 = 0.0
h2h_winrate_p1 = 0.5
h2h_days_since_last = np.nan
```

This lets the predictor operate on hypothetical pairings.

---

# 6. Score reconstruction implementation

Models do not directly output valid sport scores.

They output:

- predicted diff
- predicted total

These must be mapped back into legal scorelines.

---

## 6.1 Raw reconstruction

```python
s1 = 0.5 * (pred_total + pred_diff)
s2 = 0.5 * (pred_total - pred_diff)
```

This gives unconstrained sport scores.

---

## 6.2 Full-game decoder

Function:

```python
decode_full_game_score(pred_diff, pred_total)
```

Used for:

- TT
- BD
- SQ
- optionally TN in independent mode

It forces the sport into:

```text
21-x  or  x-21
```

using a blend of:

- reconstructed loser score
- loser score implied by margin

---

## 6.3 Tennis decoder

Function:

```python
decode_tennis_score(pred_diff, pred_total, running_diff_before_tn)
```

Supports racketlon stop-rule logic.

The function computes how many points the trailing player needs before tennis starts and truncates the tennis score when the match becomes mathematically decided.

---

# 7. Packaging / serialization

Both Tier 1 and Tier 2 save packaged predictors.

For Tier 2, the main bundle contains:

- `state_dicts`
- `feature_cols`
- `player_to_idx`
- per-sport normalization stats
- architecture config
- metrics

Serialized file:

```text
player_embedding_package.pt
```

Separate snapshot of processed feature table:

```text
data_snapshot.pkl
```

This is loaded so inference can reconstruct synthetic rows from the latest player states.

---

# 8. Predictor API

The low-level public interface is:

```python
predictor = PredictorPackage.load(path)
result = predictor.predict_pair(player1, player2)
```

The returned object is a dictionary of the form:

```python
{
    "player1": ...,
    "player2": ...,
    "sports": {
        "TT": {
            "score_p1": ...,
            "score_p2": ...,
            "pred_diff": ...,
            "pred_total": ...
        },
        ...
    },
    "total_p1": ...,
    "total_p2": ...,
    "total_diff": ...,
    "winner": ...
}
```

This format is intentionally simple so it can be used by:

- CLI scripts
- notebooks
- future web UIs
- APIs

---

# 9. Training artifacts

Training scripts save:

- per-sport scatter plots
- train/validation loss curves
- match-level scatter plots
- winner confusion matrix
- metrics JSON
- demo prediction JSON

These are generated for analysis and debugging rather than required inference.

---

# 10. Current implementation assumptions

## Assumption 1

A player’s latest row in `data.csv` is a valid summary of their latest known state.

## Assumption 2

Pair H2H can be approximated using the latest direct pair row if available.

## Assumption 3

Chronological split is more important than random IID evaluation.

## Assumption 4

Per-sport modeling is preferable to one fully joint output model.

---

# 11. Files and roles

## Feature generation

```text
features_v4.py
```

Builds `data.csv`.

## Baseline

```text
baseline.py
```

Simple ridge baseline on minimal running features.

## Tier 1

```text
catboost_regressor.py
```

CatBoost on engineered tabular features.

## Tier 2

```text
player_embedding_fin.py
```

PyTorch player-embedding MLP training and packaging.

## Tier 2 inference script

```text
player_embedding_predict.py
```

Small front-end wrapper around `PredictorPackage.load(...)`.

---

# 12. Main implementation ideas to preserve

If this project is extended, the most important implementation constraints to preserve are:

1. **No feature leakage**
2. **Chronological feature construction**
3. **Synthetic inference for unseen pairings**
4. **Clear separation between**
   - raw history
   - engineered features
   - trained model package

5. **Stable serialized predictor interface**

---

If you want, I can turn this into a polished `README.md` file next, with proper markdown formatting and section hierarchy exactly ready to paste into the repo.
