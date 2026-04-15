# Racketlon Match Prediction Project Report

## 1. Project Overview

The goal of this project was to build an end-to-end machine learning pipeline for predicting racketlon match outcomes from historical tournament data. Racketlon combines four sports in sequence:

- table tennis (TT)
- badminton (BD)
- squash (SQ)
- tennis (TN)

A racketlon match is scored by summing points across all four sports, so the modeling task naturally has both:

- **per-sport prediction**
- **overall match prediction**

This project was built as a complete pipeline:

1. scrape historical tournament results from the web
2. clean and normalize the raw matches
3. build a leakage-safe feature dataset
4. train predictive models
5. save a final inference package
6. expose reusable functions for prediction and player state lookup
7. explore geometric confidence estimation using KD-trees

The final production model is a **CatBoost-based per-sport regressor stack** that predicts:

- sport score difference
- sport total points

and then reconstructs legal scorelines and full-match outcomes.

---

# 2. Problem Definition

The main prediction target is the **total match point differential**:

[
y_{\text{total diff}} = (TT + BD + SQ + TN)*{p1} - (TT + BD + SQ + TN)*{p2}
]

The pipeline also predicts for each sport:

- `sport_y_diff`: point differential in that sport
- `sport_y_total`: total points scored in that sport

This formulation is useful because racketlon is not just a winner/loser problem. Predicting the score margin is richer and allows:

- winner prediction
- realistic reconstructed scorelines
- finer player comparisons
- model confidence analysis

The model therefore solves a structured regression problem, not just classification.

---

# 3. Data Collection

## 3.1 Source

Historical match data was scraped from Tournament Software / FIR tournament pages. The scraper handled two different site layouts:

- **new mode**
  - URLs like `/tournament/<GUID>/matches/YYYYMMDD`

- **legacy mode**
  - URLs like `/sport/legacymatches.aspx?id=<GUID>&d=YYYYMMDD`

This was important because tournament pages were not uniform across all years and events.

## 3.2 Scraper Design

The scraper:

- loaded tournament IDs from `tournament_ids.csv`
- iterated tournament-by-tournament
- used resume state via `scraper_state.json`
- retried failed requests with exponential backoff
- wrote all matches into a single `matches.csv`

### Scraped fields

For each match, the scraper collected:

- tournament ID
- date
- time
- draw
- round
- duration
- location
- player names
- player IDs if available
- nationalities and club IDs if available
- winner side
- TT/BD/SQ/TN per-player scores
- raw score text
- optional H2H link

The scraper therefore created a unified raw match table across different tournament page formats.

---

# 4. Data Cleaning

The raw scraped data was filtered and normalized before modeling.

## 4.1 Match filtering

The cleaning step removed:

- doubles and mixed events
- league/team-style draws
- incomplete matches with no raw score text
- rows without match date
- rows without squash score

The squash requirement effectively guaranteed the match was a real racketlon match rather than a partial or malformed entry.

## 4.2 Name normalization

Player names were normalized by:

- removing seeding markers like `[3/4]`
- stripping whitespace
- lowercasing all names

This was critical because player identity consistency drives all rating and historical feature calculations.

## 4.3 Output

The cleaner produced:

- `data/matches_cleaned.csv`

This file became the canonical cleaned match history.

---

# 5. Feature Engineering

The feature-building stage created:

- `data/data.csv`
- `data/inference_state.pkl`

This was the most important part of the pipeline because all models depended on leakage-safe pre-match representations.

## 5.1 Leakage-safe setup

For each match row, features were computed **before** updating any player states with the result of that match.

This means every feature corresponds to information that would have been known before the match happened.

That makes the dataset valid for real prediction rather than retrospective leakage.

---

## 5.2 Core rating model

The main internal player representation is a **single recency-sensitive rating per player per sport**.

So each player has:

- TT rating
- BD rating
- SQ rating
- TN rating

There is no separate snapshot rating system in the final pipeline.

### Rating state

For each player and sport, the system stores:

- current rating
- number of games played
- last match datetime

### Rating prediction mapping

The raw rating difference is converted into an expected point differential using:

[
\text{pred_diff} = 21 \cdot \tanh\left(\frac{\alpha x}{2}\right)
]

where:

- (x = R*{p1} - R*{p2})
- (\alpha) is sport-specific

### Sport-specific alphas

- TT: `0.060`
- BD: `0.080`
- SQ: `0.060`
- TN: `0.040`

This tanh transform is important because it:

- keeps predicted differentials bounded
- produces realistic saturation behavior
- avoids impossible margins

### Rating update rule

After a match, each sport rating is updated using:

- prediction error
- derivative of the tanh mapping
- sport-specific margin multiplier
- experience-based learning rate
- time-based update multiplier
- L2 shrinkage on the rating difference step

#### Learning rate schedule

[
\eta(g) = \eta_{\min} + (\eta_{\max} - \eta_{\min}) e^{-g/\tau}
]

Parameters:

- `ETA_MIN = 0.04`
- `ETA_MAX = 0.30`
- `ETA_TAU = 20.0`

This gives larger updates early in a player’s history and smaller updates later.

#### Margin multipliers

- TT: `0.6`
- BD: `0.8`
- SQ: `0.6`
- TN: `0.4`

Larger actual margins create larger rating updates.

#### Time-based update scaling

If a player has not played recently, the next result is allowed to move the rating more.

This does **not** shrink the stored rating itself. Instead, it increases the update size after inactivity.

Time multiplier parameters:

- TT: `TIME_C = 0.6`, `TIME_TAU_DAYS = 90`
- BD: `TIME_C = 0.7`, `TIME_TAU_DAYS = 120`
- SQ: `TIME_C = 0.6`, `TIME_TAU_DAYS = 90`
- TN: `TIME_C = 0.5`, `TIME_TAU_DAYS = 120`

#### L2 regularization in the rating update

- `L2 = 1e-5`

This is a very light stabilizer on the rating-difference step.

---

## 5.3 Recent-form features

For each player and sport, rolling windows were maintained over previous matches.

Tracked windows included:

- last 5 diffs
- last 10 diffs
- last 20 diffs
- last 5 totals
- last 10 totals
- last 5/10/20 wins
- residuals versus expected rating prediction
- blowout and close-match indicators
- favorite/underdog behavior
- exponentially weighted moving average

Although the final reduced `data.csv` keeps only the most useful subset, the internal feature generator computes a rich history.

### Important recent features retained

The final reduced feature set keeps especially:

- `diff_mean_10`
- `resid_mean_10`
- `diff_std_10`
- `momentum_diff_5_20`

These were the most useful recent-form signals in later modeling.

### Why these matter

- `diff_mean_10` captures recent raw form
- `resid_mean_10` captures performance relative to rating expectation
- `diff_std_10` captures volatility
- `momentum_diff_5_20` captures short-term directional form

---

## 5.4 Long-term shrunk features

For each player and sport, long-term aggregates were tracked:

- total matches
- total diff sum
- total total-points sum
- total wins

These were converted into shrunk estimates using:

[
\text{estimate} = \frac{\text{sum}}{n + \text{shrink}}
]

with:

- `LONG_TERM_SHRINK = 10.0`

The resulting features are:

- `long_n`
- `long_diff_mean`
- `long_total_mean`
- `long_winrate`

This shrinkage helps stabilize estimates for players with small sample sizes.

---

## 5.5 Head-to-head features

The project maintained both:

- overall H2H
- sport-specific H2H

For each pair of players, the following were tracked:

- number of prior meetings
- average differential from one player’s perspective
- win rate
- days since last meeting

These features capture matchup effects that simple ratings miss.

---

## 5.6 Targets

For each sport:

- `TT_y_diff`, `BD_y_diff`, `SQ_y_diff`, `TN_y_diff`
- `TT_y_total`, `BD_y_total`, `SQ_y_total`, `TN_y_total`

And overall:

- `y_total_diff`
- `y_winner_p1`

---

## 5.7 Inference state

At the end of feature construction, the pipeline saved `inference_state.pkl`.

This contains the final post-history state needed for synthetic future matchups:

- player ratings
- player recent-form summaries
- long-term summaries
- H2H summaries

That allows future predictions without rebuilding everything from scratch.

---

# 6. Modeling Approaches

Several models were explored.

## 6.1 Simple average-diff benchmark

A simple benchmark was built from player historical averages.

For each player and sport, it stored:

- average score differential
- average total points
- games played

Prediction rule:

- predicted sport diff = p1 average diff − p2 average diff
- predicted sport total = average of player totals

This benchmark was intentionally simple and served as a sanity check baseline.

### Performance

Approximate observed performance:

- match diff MAE around `16.7`
- winner accuracy around `0.66`

This was useful because it demonstrated that even a naive historical-average model captures some structure.

---

## 6.2 Ridge regression benchmark

A stronger benchmark used Ridge regression with engineered features.

### Why Ridge?

Ridge is linear, interpretable, and regularized. It provides a meaningful baseline against more complex models.

### Regularization

- `RIDGE_ALPHA = 3.0` in the later benchmark version

Ridge adds an L2 penalty:

[
\min_w |y - Xw|^2 + \alpha |w|^2
]

This shrinks coefficients and helps stabilize estimates when features are correlated.

### Role in the project

This benchmark was more than a pure “dumb baseline,” because it already used engineered features. It served as a structured linear baseline against CatBoost.

Observed performance was much closer to the nonlinear models than the simple average benchmark.

---

## 6.3 Player embedding neural model

A PyTorch model was also trained.

### Input structure

The model used:

- player embeddings
- numerical engineered features

For players (p_1, p_2), the model embedded both players and concatenated:

- (e_1)
- (e_2)
- (e_1 - e_2)
- (e_1 \odot e_2)
- numeric features

### Architecture

- embedding dimension: `16`
- hidden layers: `[128, 64]`
- dropout: `0.15`

### Training settings

- epochs: `80`
- batch size: `512`
- learning rate: `1e-3`
- weight decay: `1e-4`
- optimizer: `AdamW`
- loss: `SmoothL1Loss`
- early stopping patience: `10`

### Interpretation

This model tried to learn latent player skill interactions directly, rather than relying only on manually designed features.

Its performance was respectable but did not exceed the CatBoost system.

---

# 7. Final CatBoost Model

The final deployed model is a **stacked per-sport CatBoost regression system**.

This was the best practical model in the project.

## 7.1 Why CatBoost?

CatBoost was chosen because it works very well on tabular feature-engineered data:

- handles nonlinearities
- captures feature interactions
- performs well with heterogeneous numeric features
- usually needs less manual scaling/tuning than neural nets on this kind of dataset

---

## 7.2 Model structure

For each sport, three regressors are trained:

1. **base diff model**
2. **residual diff model**
3. **total model**

So each sport prediction is decomposed as:

### Base diff model

Predicts the main sport differential from stable matchup features.

### Residual diff model

Predicts the remaining error after the base model.

If base prediction is (b), residual model predicts (r), then raw diff prediction is:

[
\hat{d}_{raw} = b + 0.70r
]

where:

- `RESIDUAL_SCALE = 0.70`

This prevents the residual model from overcorrecting too aggressively.

### Diff calibrator

A linear calibrator is fit on training predictions:

[
\hat{d} = a + b \hat{d}_{raw}
]

with clipped parameters:

- slope `b` clipped to `[0.7, 1.8]`
- intercept `a` clipped to `[-5.0, 5.0]`

This fixes under/over-dispersion in the raw diff output.

### Total model

Predicts total points scored in that sport.

---

## 7.3 Feature groups

### Base diff features

The base diff model uses compact, stable features:

- `sport_rating_diff`
- `sport_games_diff`
- `sport_long_diff_mean_diff_p1_p2`
- `sport_long_winrate_diff_p1_p2`
- overall H2H:
  - `h2h_games`
  - `h2h_avg_diff_p1`
  - `h2h_winrate_p1`

- sport H2H:
  - `sport_h2h_games`
  - `sport_h2h_avg_diff_p1`
  - `sport_h2h_winrate_p1`

### Residual diff features

The residual model uses short-term form features:

- `sport_diff_mean_10_diff_p1_p2`
- `sport_resid_mean_10_diff_p1_p2`
- `sport_diff_std_10_diff_p1_p2`
- `sport_momentum_diff_5_20_diff_p1_p2`
- `sport_p1_recent_diff_std_10`
- `sport_p2_recent_diff_std_10`

Then two additional meta-features are added:

- `base_pred_diff`
- `abs_base_pred_diff`

### Total model features

The total model uses:

- rating difference
- game-count difference
- time multiplier difference
- overall and sport H2H
- recent form and residual stats
- long-term means/winrates
- long-term counts

---

## 7.4 CatBoost hyperparameters

### Base diff model

```python
iterations=900
depth=5
learning_rate=0.03
loss_function="RMSE"
eval_metric="RMSE"
random_seed=12
verbose=False
l2_leaf_reg=6
```

### Residual diff model

```python
iterations=700
depth=5
learning_rate=0.03
loss_function="RMSE"
eval_metric="RMSE"
random_seed=12
verbose=False
l2_leaf_reg=6
```

### Total model

```python
iterations=800
depth=6
learning_rate=0.03
loss_function="MAE"
eval_metric="MAE"
random_seed=12
verbose=False
l2_leaf_reg=6
```

---

## 7.5 Regularization in CatBoost

The main CatBoost regularization used was:

- `l2_leaf_reg = 6`

This is CatBoost’s leaf-value L2 regularizer. It discourages overly large leaf values and stabilizes the trees.

Other regularization effects come from:

- limited tree depth (`5` or `6`)
- moderate learning rate (`0.03`)
- staged residual correction rather than one huge model
- residual scaling (`0.70`)
- linear calibration clipping

So the final system is regularized at several levels, not just one.

---

## 7.6 Sample weighting

The project also used target-dependent sample weighting.

### Diff weights

Higher-magnitude differentials get more weight:

- `+0.15` if `abs(diff) >= 5`
- `+0.30` if `abs(diff) >= 8`
- `+0.45` if `abs(diff) >= 12`

This helps the model better represent stronger margins rather than collapsing toward conservative midrange predictions.

### Residual weights

Same weighting shape is used for residual correction.

### Total weights

For sport totals:

[
1 + 0.02 |y_{total} - 21|
]

This gives slightly more emphasis to totals that are farther from the typical center.

---

## 7.7 Score reconstruction

The regressors output:

- predicted sport diff
- predicted sport total

These are converted back into legal-looking scores.

### For TT/BD/SQ

The model decodes to a valid completed-game score such as:

- `21-16`
- `14-21`

The loser score is blended between:

- reconstructed raw score
- margin-based estimate

to create realistic outputs.

### For TN

There are two options:

- predict tennis independently
- use racketlon stop-rule logic

The project generally kept:

- `PREDICT_TENNIS_INDEPENDENTLY = True`

for simpler stable modeling.

---

# 8. Final Model Performance

Observed CatBoost performance was roughly:

- **match-level total diff MAE:** around `13.8` in evaluation splits
- **winner accuracy:** around `0.735`

In some later confidence-experiment runs where prediction was evaluated through the saved package on filtered subsets, observed overall MAE appeared lower, around `8.27`, because of the particular subset/evaluation setup used there.

The key point is that CatBoost was the strongest overall practical model in the project, especially when judged qualitatively on realistic matchup predictions.

---

# 9. Confidence Experiments with KD-Trees

A separate line of work explored whether computational geometry ideas could provide confidence estimates.

## 9.1 Motivation

Since the course emphasized computational geometry ideas such as KD-trees, the project examined whether local geometric neighborhoods could be used for confidence scoring.

The idea was:

- represent a match as a point in feature space
- query nearby historical matches with a KD-tree
- use neighborhood structure to estimate confidence

## 9.2 Experiments

### Experiment 1

Used density-only confidence in 4D rating-diff space.

Result: density was not useful and even behaved in the wrong direction.

### Experiment 2

Used local outcome consistency in the same 4D space with KNN prediction.

Result: local consistency was a meaningful confidence signal.

### Experiment 3

Applied 4D geometric confidence to CatBoost predictions.

Result: essentially no useful correlation with CatBoost error.

### Experiment 4

Expanded the KD-tree feature space to include recent and residual statistics.

Result: slight improvement, but still only a weak confidence signal.

## 9.3 Interpretation

These experiments showed:

- geometry-based confidence can work when predictor and confidence live in the same local feature space
- it becomes much weaker when confidence is computed in a feature space that is too simple relative to a stronger model like CatBoost

This became one of the main project insights.

---

# 10. Model Packaging and Inference

The final model package contains:

- trained CatBoost models for each sport
- feature column lists
- diff calibrators
- `inference_state.pkl`
- metadata JSON

A `PredictorPackage` abstraction exposes:

- `predict_pair(player1, player2)`
- `get_player_state(player)`

This allows prediction for future matches without retraining.

---

# 11. API / Service Layer

A `funcs.py` service layer was designed so the project can support a Flask app or similar interface.

Useful pipeline functions include:

- scrape matches
- clean matches
- build feature dataset
- train full model
- load saved model
- predict matchup
- fetch player rating/state
- rebuild entire pipeline end-to-end

This makes the project usable as:

- a research notebook pipeline
- a local prediction tool
- a web-backed API

---

# 12. Strengths of the Project

## End-to-end system

This is not just a model notebook. It is a full pipeline from web scraping to deployable inference.

## Leakage-safe feature construction

All features are generated using only pre-match information.

## Sport-specific structure

The project respects racketlon’s four-sport structure rather than collapsing immediately into one crude label.

## Strong tabular modeling

CatBoost is an excellent fit for this type of feature-engineered sports data.

## Rich player-state representation

The rating + recent-form + long-term + H2H combination captures multiple kinds of signal.

## Computational geometry extension

The KD-tree confidence work gives the project a meaningful research angle beyond standard supervised learning.

---

# 13. Limitations

## Missing latent factors

The model does not explicitly observe:

- injuries
- venue effects
- tournament fatigue
- surface/context effects
- travel
- age or time-varying long-term development beyond match history

## Name-based identity

Unless IDs are used, player identity depends on normalized names, which can still be imperfect.

## Tennis stop-rule simplification

The project often predicts tennis independently rather than enforcing full racketlon stop-rule dependence.

## Confidence signal mismatch

KD-tree confidence is only modestly useful for CatBoost because the geometric space is simpler than the predictor.

---

# 14. Final Conclusion

This project built a complete machine learning system for racketlon match prediction starting from raw scraped tournament data.

The pipeline:

- scrapes results across multiple site formats
- cleans and normalizes matches
- constructs leakage-safe pre-match player features
- learns sport-specific player ratings with recency-sensitive updates
- builds recent-form, long-term, and H2H features
- trains several models
- packages a final CatBoost inference system for future predictions

Among the models tested, the final **CatBoost base + residual + total** architecture gave the best overall balance of:

- predictive accuracy
- realism of outputs
- robustness
- deployability

The project also explored confidence scoring using KD-tree neighborhoods, showing that geometric confidence can work in local KNN settings but transfers only weakly to stronger nonlinear predictors unless the feature space is carefully aligned.

Overall, the project demonstrates a strong combination of:

- data engineering
- sports modeling
- feature design
- nonlinear regression
- model packaging
- computational geometry experimentation

---

# 15. Short Technical Appendix: Exact Final CatBoost Settings

## Base diff model

- iterations: `900`
- depth: `5`
- learning rate: `0.03`
- loss: `RMSE`
- eval metric: `RMSE`
- random seed: `12`
- `l2_leaf_reg = 6`

## Residual diff model

- iterations: `700`
- depth: `5`
- learning rate: `0.03`
- loss: `RMSE`
- eval metric: `RMSE`
- random seed: `12`
- `l2_leaf_reg = 6`

## Total model

- iterations: `800`
- depth: `6`
- learning rate: `0.03`
- loss: `MAE`
- eval metric: `MAE`
- random seed: `12`
- `l2_leaf_reg = 6`

## Other modeling choices

- residual scale: `0.70`
- diff calibrator slope clip: `[0.7, 1.8]`
- diff calibrator intercept clip: `[-5.0, 5.0]`
- tennis independent prediction: `True`
