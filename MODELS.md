# Racketlon Predictor — Model and Rating Implementation Notes

This document describes the core modeling pipeline at the implementation level:

1. **Ratings**
2. **Baseline model**
3. **CatBoost model**
4. **Player embedding model**

The focus is on:

- what each component is
- how it is computed
- why it behaves the way it does

---

# 1. Ratings

The project uses **two different notions of rating**:

- **current running rating**
- **snapshot rating**

These are not the same thing.

---

## 1.1 Current running rating

This is the dynamic, match-by-match rating used in the feature pipeline.

It is implemented in the `MarginElo` class and updated **sequentially in chronological order**.

### Per-player state

For each player and each sport, the system stores:

```python
self.R[player][sport]
self.games[player][sport]
```

Where:

- `R[player][sport]` = current rating
- `games[player][sport]` = number of completed matches in that sport

The sports are:

```python
SPORTS = ["TT", "BD", "SQ", "TN"]
```

---

## 1.2 Current rating prediction function

Before updating ratings, the system computes the expected score difference from the rating difference.

For a sport:

```python
x = R[p1][sport] - R[p2][sport]
pred = 21 * tanh((alpha * x) / 2)
```

Implementation:

```python
def rating_to_score_diff(x: float, alpha: float) -> float:
    return MAX_DIFF * math.tanh((alpha * x) / 2.0)
```

Where:

- `MAX_DIFF = 21`
- `alpha` is a fixed slope parameter per sport in this running-rating system

Typical values used:

```python
ALPHAS = {"TT": 0.010, "BD": 0.010, "SQ": 0.010, "TN": 0.010}
```

This means:

- if rating difference is small, expected score difference is small
- if rating difference is large, expected score difference saturates toward ±21

---

## 1.3 Current rating update rule

After the actual match is observed, the rating is updated using prediction error.

### Step 1: compute actual diff

```python
d = actual_score_p1 - actual_score_p2
```

### Step 2: compute prediction error

```python
err = d - pred
```

### Step 3: compute learning rate

The learning rate decays with the number of games played:

```python
eta_eff(games_played) = ETA_MIN + (ETA_MAX - ETA_MIN) * exp(-games_played / ETA_TAU)
```

with:

```python
ETA_MIN = 0.02
ETA_MAX = 0.20
ETA_TAU = 15.0
```

Then the two-player learning rate is averaged:

```python
eta = 0.5 * (eta_eff(g1) + eta_eff(g2))
```

This means:

- newer players update faster
- established players update more slowly

---

## 1.4 Current rating gradient step

The predicted diff is nonlinear in the rating difference, so the update uses the derivative:

```python
grad = d_pred_dx(x, alpha)
step = eta * err * grad
step -= eta * L2 * x
```

with small regularization:

```python
L2 = 1e-5
```

Then ratings are updated symmetrically:

```python
R[p1] = r1 + step
R[p2] = r2 - step
```

This makes it:

- zero-sum between the two players
- stable over time
- sensitive to prediction error

---

## 1.5 What current ratings represent

These ratings are intended to capture:

> **recent form and sequentially updated strength**

Because they are updated after every match, they move faster than monthly snapshots.

Stored feature columns include:

```text
TT_rating_p1
TT_rating_p2
TT_rating_diff
TT_games_p1
TT_games_p2
TT_games_diff
```

and similarly for BD, SQ, TN.

---

# 2. Snapshot ratings

Snapshot ratings are slower, more stable ratings computed on a **monthly basis**.

They are designed to represent:

> **long-term player strength at a fixed point in time**

---

## 2.1 Why snapshots exist

The current running rating is:

- dynamic
- order-sensitive
- more reactive

The snapshot rating is:

- slower
- more stable
- less noisy
- more like a monthly skill estimate

This gives the model both:

- short-term form
- long-term skill

---

## 2.2 Snapshot computation logic

Snapshots are computed by grouping matches by month, then iteratively updating a monthly rating table.

At a high level:

1. sort matches chronologically
2. group them by `(year, month)`
3. for each month:
   - run rating updates over matches in that month
   - store resulting player ratings as the month snapshot

Internally, the monthly updater uses:

- player history initialization
- decayed weighting by recency
- repeated passes over the month’s matches

---

## 2.3 Initialization from player history

If a player does not yet have a monthly rating entry, the system initializes it using historical average margin over a long window.

Implementation idea:

```python
compute_history(player, date, history, within=100)
```

This computes average score margins for each sport over the player’s past matches.

So the initial monthly rating is not random or zero-only; it is seeded from actual historical performance.

---

## 2.4 Monthly update rule

For each month, ratings are updated repeatedly over that month’s matches:

```python
for j in range(1, 10):
    update_all_ratings(rows, ratings_by_month, history, ref_date, base_eta=1 / j)
```

This is effectively a smoothing procedure:

- earlier passes allow larger movement
- later passes refine ratings with smaller steps

The update logic compares:

```python
actual_diff - expected_diff
```

where expected diff is the difference between players’ monthly ratings.

---

## 2.5 Snapshot alpha fitting

In the older snapshot system, alphas were fit separately per sport to map rating diff to expected score diff.

This logic existed primarily for the older rating-only model and for generating transformed predicted-diff features.

Later, for CatBoost, the raw snapshot ratings turned out to be more important than these transformed features.

Stored snapshot columns include:

```text
TT_snapshot_rating_p1
TT_snapshot_rating_p2
TT_snapshot_rating_diff
TT_snapshot_p1_found
TT_snapshot_p2_found
```

---

## 2.6 What snapshots represent

Snapshot ratings are best interpreted as:

> **monthly frozen skill estimates**

They do not change from match to match inside the same month.

In practice, they ended up being the strongest single feature family in the project.

---

# 3. Baseline model

The baseline is a simple regression model built directly from **running averages**, without complex feature engineering.

Its purpose is to provide:

- a sanity check
- a low-capacity comparison point
- a simple benchmark before using more expressive models

---

## 3.1 Baseline feature logic

The baseline computes pre-match running averages for each player and sport.

For each player and sport, it stores:

```python
diff_sum[player][sport]
total_sum[player][sport]
count[player][sport]
```

Then the pre-match average is:

```python
avg_diff = diff_sum / (count + shrink)
avg_total = total_sum / (count + shrink)
```

with:

```python
shrink = 10
```

This shrinkage acts like a prior:

- if a player has few matches, averages are shrunk toward zero
- if a player has many matches, their actual average dominates

---

## 3.2 Baseline per-sport features

For a match between p1 and p2, the baseline computes:

```python
delta_diff = p1_avg_diff - p2_avg_diff
delta_total = p1_avg_total - p2_avg_total
sum_total = p1_avg_total + p2_avg_total
```

These are the only sport-level inputs.

So per sport, the baseline uses 3 features:

- relative average margin
- relative average total points
- combined average total points

This is a very compact representation.

---

## 3.3 Baseline model type

The baseline uses **Ridge regression**.

For each sport:

- one model predicts score difference
- one model predicts score total

Implementation pattern:

```python
model_diff = Ridge(alpha=1.0)
model_total = Ridge(alpha=1.0)
```

This means:

- linear model
- L2 regularization
- low variance
- easy to interpret

---

## 3.4 Why Ridge

Ridge is used because:

- features are small and dense
- it is stable
- it prevents coefficient blow-up
- it provides a good “simple but reasonable” baseline

It is not designed to capture:

- nonlinear interactions
- higher-order effects
- sparse conditional logic

That limitation is why Tier 1 and Tier 2 were added.

---

# 4. CatBoost model

CatBoost is the main **Tier 1** production-style model.

It is a **gradient boosted decision tree model** for tabular data.

---

## 4.1 What kind of model CatBoost is

CatBoost is a boosting algorithm that builds an ensemble of decision trees sequentially.

At a high level:

1. start with a simple prediction
2. compute residuals / errors
3. fit a new tree to reduce those errors
4. add that tree into the ensemble
5. repeat many times

So instead of learning one global linear function, it learns:

> **a sum of many small decision trees**

This makes it strong for:

- nonlinear tabular data
- heterogeneous feature scales
- interactions between features
- missing values

---

## 4.2 CatBoost model decomposition

Like the baseline, CatBoost is trained per sport.

For each sport:

- one regressor predicts `*_y_diff`
- one regressor predicts `*_y_total`

So total CatBoost models trained:

```text
4 sports × 2 targets = 8 regressors
```

This keeps each target simpler.

---

## 4.3 CatBoost feature set

The model is trained on engineered tabular features from `data.csv`.

Important included groups:

- current ratings
- snapshot ratings
- H2H
- games played
- global H2H
- snapshot found flags

Important excluded groups:

- target columns
- `has_*`
- old transformed `*_pred_diff`
- `snapshot_total_pred_diff`
- identifiers

This filtering is deliberate to reduce:

- redundancy
- leakage risk
- dependence on legacy transforms

---

## 4.4 CatBoost parameters

Typical implementation:

```python
CATBOOST_PARAMS = dict(
    iterations=800,
    depth=6,
    learning_rate=0.03,
    loss_function="MAE",
    eval_metric="MAE",
    random_seed=12,
    verbose=False,
    l2_leaf_reg=5,
)
```

---

## 4.5 What each CatBoost parameter means

### `iterations=800`

Number of boosting rounds.

Each iteration adds another tree.

Higher iterations:

- can fit more complex patterns
- can overfit if too high

---

### `depth=6`

Maximum depth of each decision tree.

Deeper trees:

- capture more interactions
- are more expressive
- can overfit more easily

Depth 6 is a moderate setting.

---

### `learning_rate=0.03`

Shrinkage factor on each tree’s contribution.

Smaller learning rate:

- slower learning
- usually more stable
- often requires more iterations

  0.03 is a conservative value.

---

### `loss_function="MAE"`

Training objective is mean absolute error.

This makes the model optimize absolute residuals rather than squared residuals.

MAE is more robust to extreme outliers than RMSE.

Since score prediction can occasionally contain noisy or unusual results, MAE is a reasonable choice.

---

### `eval_metric="MAE"`

Validation metric also uses MAE.

So train objective and evaluation metric are aligned.

---

### `random_seed=12`

Controls reproducibility.

Important because boosted trees can depend on random feature / split decisions.

---

### `verbose=False`

Suppresses per-iteration logging.

Purely cosmetic.

---

### `l2_leaf_reg=5`

L2 regularization on leaf values.

This penalizes overly large leaf predictions and helps reduce overfitting.

Higher values:

- smoother model
- more bias
- less variance

Lower values:

- more aggressive fitting
- more variance

---

## 4.6 How CatBoost works on this project

CatBoost ended up learning primarily from:

- raw snapshot ratings
- raw current ratings
- some H2H and games played features

When snapshot-derived transformed features were removed, performance stayed strong, which showed:

> the model preferred raw player-level rating inputs over handcrafted transformed versions

That is a sign CatBoost is learning the comparison itself.

---

# 5. Player embedding model

This is the **Tier 2** deep learning model.

It combines:

- learned player representations
- engineered numeric features

It is implemented in PyTorch.

---

## 5.1 What kind of model it is

The player embedding model is a **feedforward neural network with learned embeddings**.

Per sport, it predicts:

- score difference
- score total

It is not an RNN or Transformer yet. It is a dense MLP over:

- player identity embeddings
- engineered numeric inputs

---

## 5.2 Embedding idea

Each player is assigned an integer ID:

```python
player_to_idx[player_key] = integer
```

Then the model learns an embedding vector:

```python
self.embed = nn.Embedding(n_players + 1, embed_dim)
```

So each player gets a learned dense vector of length `embed_dim`.

Typical value:

```python
EMBED_DIM = 16
```

These embeddings allow the model to capture:

- latent player skill
- style similarity
- interaction structure not explicitly encoded in features

---

## 5.3 Model input construction

For a given match, the model retrieves:

```python
e1 = embedding(player1)
e2 = embedding(player2)
```

Then forms the combined input:

```python
[e1, e2, e1 - e2, e1 * e2, numeric_features]
```

This is important:

- `e1` = player 1 latent identity
- `e2` = player 2 latent identity
- `e1 - e2` = relative latent difference
- `e1 * e2` = latent interaction term

This structure lets the network learn more than simple difference.

---

## 5.4 Numeric feature input

The embedding model also uses the same engineered numeric features as the CatBoost model:

- current ratings
- snapshot ratings
- H2H
- games
- global H2H

So it is a **hybrid model**, not an embedding-only model.

---

## 5.5 MLP architecture

Typical implementation parameters:

```python
EMBED_DIM = 16
HIDDEN_DIMS = [128, 64]
DROPOUT = 0.15
```

This means:

### Input layer

Dimension:

```text
numeric_features + 4 * EMBED_DIM
```

because of:

- `e1`
- `e2`
- `e1 - e2`
- `e1 * e2`

### Hidden layers

Two fully connected layers:

- 128 units
- 64 units

with:

- ReLU activation
- dropout after each layer

### Output head

Final linear layer outputs 2 values:

- predicted diff
- predicted total

---

## 5.6 Player embedding model training parameters

Typical settings:

```python
EPOCHS = 80
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10
```

---

## 5.7 What these mean

### `EPOCHS = 80`

Maximum full passes over the training set.

Actual training may stop earlier due to early stopping.

---

### `BATCH_SIZE = 512`

Number of training examples per optimizer step.

Large enough for stable gradients, small enough to fit comfortably.

---

### `LEARNING_RATE = 1e-3`

Step size for AdamW.

A standard safe choice for small-to-medium MLPs.

---

### `WEIGHT_DECAY = 1e-4`

L2-style regularization in AdamW.

Helps reduce overfitting in dense layers and embeddings.

---

### `EARLY_STOPPING_PATIENCE = 10`

If validation loss does not improve for 10 epochs, training stops.

This is important because the model can otherwise overfit tabular data.

---

## 5.8 Loss function

The player embedding model uses:

```python
nn.SmoothL1Loss()
```

for both outputs.

Total loss:

```python
loss = loss_diff + loss_total
```

Smooth L1 is chosen because it is:

- more robust than MSE
- smoother than pure MAE
- good for noisy regression targets

---

## 5.9 Optimization

Optimizer:

```python
torch.optim.AdamW(...)
```

AdamW is used because:

- it handles different gradient scales well
- it works well for embeddings + dense layers
- weight decay is decoupled cleanly from the optimizer

---

# 6. Predictor interface

Both CatBoost and player embedding models are wrapped in a `PredictorPackage`.

The common interface is:

```python
predictor = PredictorPackage.load(path)
result = predictor.predict_pair(player1, player2)
```

This is important because it decouples:

- training implementation
- saved model format
- downstream usage

---

# 7. Summary of model roles

## Ratings

- current rating = dynamic sequential form
- snapshot rating = stable monthly skill

## Baseline

- avg-based
- linear ridge regression
- minimal feature set
- sanity-check benchmark

## CatBoost

- gradient boosted decision tree ensemble
- nonlinear tabular learner
- strong on engineered structured features
- uses explicit regularization via tree depth, shrinkage, and `l2_leaf_reg`

## Player embeddings

- neural network with learned player vectors
- combines identity embeddings with engineered numeric features
- more flexible representation learning
- higher tuning burden than CatBoost
