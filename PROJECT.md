# Racketlon Match Prediction Project Report

## Abstract

This project develops an end-to-end machine learning pipeline for predicting racketlon match outcomes from historical tournament data. Racketlon is a four-sport sequence consisting of table tennis, badminton, squash, and tennis, where the winner is determined by the total number of points accumulated across all four sports. The project begins with automated web scraping of tournament results from multiple website formats, continues through cleaning and normalization of raw match records, and then constructs a leakage-safe feature dataset designed for pre-match prediction. Several modeling approaches were explored, including simple historical-average baselines, linear regularized regression, neural models with player embeddings, and a final production system based on gradient-boosted decision tree regressors. The final model predicts both sport-level score differences and sport-level total points, reconstructs legal scorelines, and aggregates these predictions into full match outcomes. In addition to prediction, the project also investigates confidence estimation using computational geometry ideas based on KD-trees, motivated by course themes. These confidence experiments test whether local neighborhood density and local outcome consistency can serve as proxies for predictive reliability. The final result is a reusable, deployable prediction system that combines data engineering, dynamic player rating updates, feature engineering, nonlinear supervised learning, and geometric analysis. The project demonstrates how a real-world sports analytics pipeline can be built from scratch, from raw web data to production-ready inference functions, while also exploring research-style questions about model confidence and feature-space geometry.

---

## 1. Introduction

The objective of this project was to build a machine learning system capable of predicting racketlon match outcomes from historical data. Racketlon is particularly interesting as a prediction task because it is not a single-sport classification problem. Instead, each match consists of four sports played in order:

- Table Tennis (TT)
- Badminton (BD)
- Squash (SQ)
- Tennis (TN)

The final winner is determined by the total number of points won across all four sports. This creates a structured prediction problem with multiple levels:

1. Predict the relative advantage of one player over another in each individual sport.
2. Predict realistic sport scorelines.
3. Aggregate across sports to obtain a full match prediction.
4. Predict the winner and total match margin.

This makes racketlon more complex than a binary sports prediction task. A good model must capture both broad player strength and sport-specific specialization. It must also account for short-term form, long-term history, and matchup-specific information.

The project was designed as a full pipeline rather than a single isolated model. It includes:

- automated data collection from tournament websites
- cleaning and validation of raw scraped results
- leakage-safe feature engineering
- dynamic player rating estimation
- multiple baseline and advanced models
- saved inference packages for future matchup prediction
- confidence experiments using KD-tree neighborhood geometry

In addition to predictive performance, a second goal was to produce a system that could be called from a service layer or web app. For that reason, the final system was organized into reusable functions for scraping, cleaning, feature building, model training, loading, and prediction.

---

## 2. Problem Definition

### 2.1 Prediction targets

At the sport level, for each sport $s \in \{TT, BD, SQ, TN\}$, the project predicts:

- sport point differential:
  $y^{(s)}_{\text{diff}} = s_{p1} - s_{p2}$

- sport total points:
  $y^{(s)}_{\text{total}} = s_{p1} + s_{p2}$

At the match level, the total score differential is:

$y_{\text{total diff}} = \sum_{s}(s_{p1} - s_{p2})$

The winner target is then:

$y_{\text{winner}} = \mathbf{1}(y_{\text{total diff}} > 0)$

This structure was chosen because directly predicting only the winner would throw away a large amount of useful information. Predicting the score differential and sport totals supports:

- winner prediction
- realistic score reconstruction
- better player diagnostics
- finer evaluation metrics
- interpretable predictions

### 2.2 Why regression instead of just classification

Although winner prediction is useful, regression is more informative in this setting. If two players are predicted to be close overall, the model should be able to show _where_ that closeness comes from. Similarly, if one player is expected to dominate badminton but lose badly in table tennis, the model should express that directly.

Therefore, the core problem is treated as a structured regression task rather than a pure classification task.

---

## 3. Data Collection

## 3.1 Source website structure

Historical results were collected from Tournament Software / FIR tournament pages. A complication in the project was that the site exists in multiple layouts depending on tournament age and page version. Two major formats had to be supported:

1. **New mode**
   URLs of the form:
   `/tournament/<GUID>/matches/YYYYMMDD`

2. **Legacy mode**
   URLs of the form:
   `/sport/legacymatches.aspx?id=<GUID>&d=YYYYMMDD`

Because older tournaments and newer tournaments used different HTML structures, the scraper had to be hybrid rather than assuming one fixed page layout.

## 3.2 Tournament ID loading

The scraper begins from a file `tournament_ids.csv` that stores tournament GUIDs. These IDs act as the root input to the scraping system. Each tournament ID is normalized and processed in turn.

## 3.3 Resumable scraping

To make large-scale scraping reliable, the script uses a persistent state file:

- `scraper_state.json`

This file records completed tournaments so that if scraping is interrupted, the script can resume from where it left off. This avoids re-downloading previously processed tournaments and makes the pipeline robust for long-running jobs.

## 3.4 Request handling and retries

The scraper uses HTTP requests with:

- explicit headers
- timeout handling
- retry logic
- exponential backoff

This is important because tournament websites occasionally fail temporarily or throttle requests.

## 3.5 New-mode scraping logic

In the newer tournament format, the scraper:

1. opens the tournament matches page
2. discovers day-specific URLs
3. fetches each daily page
4. extracts each match from `div.match.match--list`
5. parses metadata such as:
   - match time
   - draw
   - round
   - location
   - players
   - winner side
   - sport-by-sport points
   - optional head-to-head URL

The parser also recovers player IDs, nationality IDs, and club IDs when available.

## 3.6 Legacy-mode scraping logic

In the older format, the scraper:

1. discovers day URLs from the tournament page calendar
2. opens each legacy day page
3. identifies the most likely match table
4. builds a flexible header mapping
5. parses score cells and player columns
6. reconstructs TT/BD/SQ/TN scores from cell text

Legacy pages are less standardized, so the parser uses more robust fallback rules.

## 3.7 Unified raw output

Regardless of source mode, both scraping branches produce the same row schema. This unified schema includes:

- `mode`
- `tournament_id`
- `match_date`
- `match_time`
- `draw`
- `draw_id`
- `round`
- `duration`
- `location`
- `team1_players`
- `team2_players`
- player IDs
- nationalities
- club IDs
- `winner_side`
- `status_message`
- `TT_p1`, `TT_p2`
- `BD_p1`, `BD_p2`
- `SQ_p1`, `SQ_p2`
- `TN_p1`, `TN_p2`
- `raw_points`
- `h2h_url`

These rows are appended into `matches.csv`.

---

## 4. Data Cleaning

The raw scraped data is not directly suitable for modeling. Some rows correspond to irrelevant events or malformed records. The cleaning stage creates a stricter racketlon-only dataset.

## 4.1 Removed rows

The cleaning script removes matches that satisfy any of the following:

- draw names containing invalid terms such as:
  - `double`
  - `doubles`
  - `mixed`
  - `league`
  - `team`

These usually indicate formats that are not standard singles racketlon matches.

- rows with missing `raw_points`
- rows with missing `match_date`
- rows missing squash scores

The squash requirement was especially useful because valid racketlon matches should include all four sports, and missing squash often indicated incomplete or corrupted records.

## 4.2 Name normalization

Player names are normalized by:

- removing seeding markers such as `[3]` or `[3/4]`
- trimming whitespace
- lowercasing all text

This is crucial because the feature engineering pipeline relies heavily on consistent player identity.

If `"Zain Magdon-Ismail"` and `"zain magdon-ismail [3]"` were treated as different players, the rating and history system would break down.

## 4.3 Cleaned output

The cleaner produces:

- `data/matches_cleaned.csv`

This file is the canonical cleaned historical match dataset.

---

## 5. Feature Engineering Pipeline

Feature generation is handled by `features.py`. This script performs the main stateful transformation from cleaned raw matches to a training-ready dataset.

It produces two outputs:

- `data/data.csv`
- `data/inference_state.pkl`

The feature generator is designed around one critical principle:

> Every feature for match $i$ must be computed only from matches that occurred before match $i$.

That makes the dataset **leakage-safe**.

---

## 5.1 Core state objects

The feature pipeline maintains several kinds of evolving state as it iterates chronologically through matches:

1. **Per-player per-sport ratings**
2. **Per-player per-sport recent-form windows**
3. **Per-player per-sport long-term aggregates**
4. **Pairwise overall head-to-head state**
5. **Pairwise sport-specific head-to-head state**

Each match row is processed in order:

1. read current states
2. generate features
3. record targets
4. update states using the match outcome

This order guarantees correctness for future prediction.

---

## 5.2 Time handling

The helper `safe_datetime()` constructs a usable chronological timestamp from:

- `datetime` if already present
- otherwise `match_date`
- optionally `match_time`

This allows matches to be ordered correctly and enables time-based features such as inactivity.

---

## 5.3 Dynamic one-rating-per-sport system

A central part of the project is the custom rating model `DecayedUpdateMarginElo`.

Each player has one scalar rating per sport:

- $R_{p,TT}$
- $R_{p,BD}$
- $R_{p,SQ}$
- $R_{p,TN}$

This is a compact, sport-specific latent strength representation.

### 5.3.1 Rating-to-score mapping

A raw rating difference is converted to an expected point differential using:

$\hat{d} = 21 \cdot \tanh\left(\frac{\alpha x}{2}\right)$

where:

- $x = R_{p1,s} - R_{p2,s}$
- $\alpha$ is sport-specific
- the output is bounded between $-21$ and $21$

This mapping has several advantages:

- it prevents impossible raw predicted margins
- it saturates naturally for large rating gaps
- it behaves almost linearly near zero
- it is sport-specific through $\alpha$

### 5.3.2 Alpha values

The values used were:

- TT: `0.060`
- BD: `0.080`
- SQ: `0.060`
- TN: `0.040`

These control how rapidly rating differences translate into score expectations.

Higher $\alpha$ means smaller rating differences produce stronger expected score differentials.

### 5.3.3 Learning rate schedule

The update size depends on experience:

$\eta(g) = \eta_{\min} + (\eta_{\max} - \eta_{\min}) e^{-g/\tau}$

Parameters:

- $\eta_{\min} = 0.04$
- $\eta_{\max} = 0.30$
- $\tau = 20.0$

Interpretation:

- early in a player's history, ratings move quickly
- after many matches, ratings stabilize

### 5.3.4 Time-based update multiplier

If a player has been inactive, the next result is allowed to update the rating more strongly.

The multiplier is:

$\mathrm{time\_mult} = 1 + c_s \left(1 - e^{-d/\tau_s}\right)$

where:

- $d$ = days since last match in that sport
- $c_s$ = sport-specific maximum adjustment coefficient
- $\tau_s$ = sport-specific timescale

This does **not** decay the stored rating. Instead, it increases the sensitivity of the next update.

Parameters:

- TT:
  - $c = 0.6$
  - $\tau = 90$
- BD:
  - $c = 0.7$
  - $\tau = 120$
- SQ:
  - $c = 0.6$
  - $\tau = 90$
- TN:
  - $c = 0.5$
  - $\tau = 120$

### 5.3.5 Margin-sensitive updates

The update also scales with observed match margin:

$\text{margin\_mult} = 1 + m_s \cdot \frac{|d|}{21}$

where $m_s$ is sport-specific:

- TT: `0.6`
- BD: `0.8`
- SQ: `0.6`
- TN: `0.4`

This lets bigger wins or losses matter more than narrow results.

### 5.3.6 Gradient term

Because the prediction mapping is nonlinear, the update uses the derivative of the tanh transform:

$\frac{d\hat{d}}{dx} = 21 \cdot \frac{\alpha}{2}(1 - \tanh^2(\alpha x / 2))$

This ensures the learning signal behaves properly depending on where the players lie in rating space.

### 5.3.7 L2 regularization in rating updates

A small L2 term is applied against rating difference growth:

- `L2 = 1e-5`

This is a very mild stabilizer rather than a strong shrinkage force.

### 5.3.8 Full update intuition

The update is approximately:

$\Delta x \propto \eta \cdot \text{time\_mult} \cdot \text{margin\_mult} \cdot (d - \hat{d}) \cdot \frac{d\hat{d}}{dx}$

So the system behaves like a nonlinear error-correcting margin-sensitive Elo variant.

---

## 5.4 Recent-form feature state

The project maintains rolling history windows per player and sport using `RollingWindow`.

These track recent match sequences of various lengths:

- 5
- 10
- 20

### 5.4.1 Stored recent metrics

For each player and sport, the pipeline tracks:

- recent score differences
- recent total points
- recent wins
- blowout wins and losses
- close matches
- residuals vs expected rating prediction
- positive residual rate
- favorite/underdog behavior
- exponentially weighted moving average

### 5.4.2 Meaning of residuals

Residuals are defined as:

$\text{residual} = \text{actual diff} - \text{expected diff}$

This is useful because it measures whether a player is overperforming or underperforming relative to rating-based expectation.

### 5.4.3 Momentum

Momentum is defined as:

$\text{momentum diff}_{5,20} = \text{mean diff over last 5} - \text{mean diff over last 20}$

This gives a simple trend signal:

- positive means short-term performance is improving
- negative means form may be fading

### 5.4.4 Why rolling features matter

The rating alone is intended to capture broad medium-term strength. Recent features capture things the rating may not move fast enough to reflect, such as:

- streaks
- recent improvement
- temporary decline
- volatility
- persistent over/underperformance

---

## 5.5 Long-term shrunk features

In addition to recent windows, each player and sport maintains long-term aggregates:

- number of matches
- cumulative score difference
- cumulative total points
- cumulative wins

These are converted into shrunk estimates:

$\text{long estimate} = \frac{\text{sum}}{n + \lambda}$

with:

- $\lambda = 10.0$

This reduces instability for players with small sample sizes.

The final long-term features are:

- `long_n`
- `long_diff_mean`
- `long_total_mean`
- `long_winrate`

These help distinguish players with similar recent form but very different historical baselines.

---

## 5.6 Head-to-head features

For each player pair, the project stores both overall and sport-specific head-to-head information.

### 5.6.1 Overall H2H features

- total prior meetings
- average prior total-diff from player 1 perspective
- win rate from player 1 perspective
- days since last meeting

### 5.6.2 Sport-specific H2H features

For each sport:

- sport-specific prior meetings
- average sport differential
- sport-specific win rate
- days since last meeting in that sport context

These features are valuable because some player matchups are consistently favorable or unfavorable beyond what global ratings indicate.

---

## 5.7 Output dataset structure

The feature pipeline writes a reduced final dataset containing:

### Identity columns

- `match_index`
- `datetime`
- `month_key`
- `p1_key`
- `p2_key`
- `p1_name`
- `p2_name`

### Core sport features

For each sport:

- ratings
- games played
- inactivity
- time multipliers
- H2H features
- selected recent-form features
- selected long-term features

### Match-level H2H features

- `h2h_games`
- `h2h_avg_diff_p1`
- `h2h_winrate_p1`
- `h2h_days_since_last`

### Targets

- `sport_y_diff`
- `sport_y_total`
- `y_total_diff`
- `y_winner_p1`

This reduced dataset was designed to keep the useful features while avoiding bloated or unused columns.

---

## 5.8 Inference state

The pipeline also saves:

- `data/inference_state.pkl`

This contains the final post-history state needed to synthesize a future matchup row without recomputing the entire feature history.

It stores:

- player state summaries
- per-sport ratings
- recent-form summaries
- long-term summaries
- pairwise H2H summaries

This file is what makes future prediction possible from just two player names.

---

## 6. Modeling Approaches Explored

Several modeling approaches were tested before settling on the final production model.

---

## 6.1 Simple historical-average benchmark

This benchmark predicts sport differentials using player historical averages.

For each player and sport, it computes:

- average score differential
- average sport total
- games played

Prediction is then:

$\hat{d}^{(s)} = \bar{d}^{(s)}_{p1} - \bar{d}^{(s)}_{p2}$

And the sport total is estimated from average totals.

### Strengths

- simple
- fully interpretable
- very fast
- useful as a genuine baseline

### Weaknesses

- no nonlinear interactions
- no matchup-specific correction beyond averages
- no dynamic weighting of recent vs old form
- no learned correction structure

### Observed performance

This model achieved roughly:

- match total diff MAE around `16.7`
- winner accuracy around `0.66`

This demonstrated that historical averages already carry substantial predictive signal.

---

## 6.2 Ridge regression benchmark

A stronger baseline was built using Ridge regression.

Ridge solves:

$\min_w ||y - Xw||^2 + \alpha ||w||^2$

where:

- $X$ is the feature matrix
- $y$ is the target
- $\alpha$ controls regularization strength

### Parameter

- `RIDGE_ALPHA = 3.0`

### Why Ridge was used

Ridge is a useful baseline because:

- it is linear and interpretable
- it handles correlated features better than unregularized linear regression
- it gives a reasonable tabular baseline against nonlinear models

### Interpretation of regularization

The L2 penalty shrinks coefficient magnitudes, reducing sensitivity to noisy feature correlations and improving stability.

### Role in the project

Although it was called a benchmark, this model was stronger than a trivial baseline because it still used the engineered pre-match features. It provided a meaningful linear reference point relative to the boosted-tree system.

---

## 6.3 Player embedding neural network

A PyTorch-based neural model was also explored.

### 6.3.1 Representation

Each player is given an embedding vector $e_p$ of dimension 16.

Given players $p_1$ and $p_2$, the model forms input from:

- $e_1$
- $e_2$
- $e_1 - e_2$
- $e_1 \odot e_2$
- numeric engineered features

This allows the model to learn latent player identity effects that may not be fully captured by hand-engineered statistics.

### 6.3.2 Architecture

- embedding dimension: `16`
- hidden layers: `[128, 64]`
- dropout: `0.15`

The final head outputs two values:

- predicted sport diff
- predicted sport total

### 6.3.3 Optimization details

- optimizer: `AdamW`
- learning rate: `1e-3`
- weight decay: `1e-4`
- loss: `SmoothL1Loss`
- batch size: `512`
- epochs: `80`
- early stopping patience: `10`

### 6.3.4 Interpretation

This model tries to learn player-specific latent traits, but on this project’s tabular data, it did not surpass the final boosted-tree system.

---

## 7. Final Model: Stacked Gradient-Boosted Decision Tree Regression System

The final production model is a sport-wise ensemble of gradient-boosted decision tree regressors.

This was implemented with the CatBoost library, but methodologically the model should be described as a **gradient-boosted decision tree regression system**.

It was the strongest overall model in terms of:

- quantitative metrics
- realism of predictions
- robustness
- ease of deployment

---

## 7.1 Why gradient-boosted trees were a good fit

Gradient-boosted trees are especially effective for structured tabular data because they:

- capture nonlinear feature interactions
- handle mixed scales without explicit normalization
- learn threshold effects naturally
- perform well with medium-sized engineered datasets
- often outperform neural nets on tabular problems unless the data is extremely large or has richer raw modalities

In this project, the features were already information-dense and semantically meaningful, which is an excellent setting for boosted trees.

---

## 7.2 Per-sport decomposition

For each sport, three separate regressors are trained:

1. **base differential model**
2. **residual correction model**
3. **total-points model**

This means each sport has its own structured prediction pipeline.

### 7.2.1 Base differential model

The base model predicts the main sport differential from stable, matchup-level features.

This model is responsible for broad skill-gap prediction.

### 7.2.2 Residual correction model

The residual model learns to correct the remaining error of the base model.

If the base model prediction is $b$ and the residual model prediction is $r$, then the raw differential prediction is:

$\hat{d}_{raw} = b + \gamma r$

where:

- $\gamma = 0.70$

This residual scaling was chosen to avoid overcorrection and to keep the residual model as a refinement rather than a replacement.

### 7.2.3 Differential calibration

After base + residual combination, a linear calibrator is fit:

$\hat{d}_{final} = a + b \hat{d}_{raw}$

The calibrator parameters are clipped:

- slope $b \in [0.7, 1.8]$
- intercept $a \in [-5.0, 5.0]$

This helps correct under-dispersion or systematic bias in the raw differential model.

### 7.2.4 Total-points model

A separate model predicts total points in the sport. This is important because the same point differential can correspond to different realistic scorelines.

For example:

- `21-18` and `21-10` are both wins, but have different totals and realism implications.

---

## 7.3 Feature groups used by the final model

The final tree system intentionally used **compact feature groups** rather than every possible engineered feature.

### 7.3.1 Base differential features

The base differential model uses stable matchup descriptors:

- `sport_rating_diff`
- `sport_games_diff`
- `sport_long_diff_mean_diff_p1_p2`
- `sport_long_winrate_diff_p1_p2`
- overall H2H:
  - `h2h_games`
  - `h2h_avg_diff_p1`
  - `h2h_winrate_p1`
- sport-specific H2H:
  - `sport_h2h_games`
  - `sport_h2h_avg_diff_p1`
  - `sport_h2h_winrate_p1`

These are the broad, relatively slow-moving descriptors of player strength and matchup history.

### 7.3.2 Residual model features

The residual correction model focuses on short-term mismatch between the base estimate and recent evidence:

- `sport_diff_mean_10_diff_p1_p2`
- `sport_resid_mean_10_diff_p1_p2`
- `sport_diff_std_10_diff_p1_p2`
- `sport_momentum_diff_5_20_diff_p1_p2`
- `sport_p1_recent_diff_std_10`
- `sport_p2_recent_diff_std_10`

Two extra meta-features are added:

- `base_pred_diff`
- `abs_base_pred_diff`

These allow the residual model to condition its correction on the size and direction of the base estimate.

### 7.3.3 Total-points model features

The total model uses a slightly broader set:

- `sport_rating_diff`
- `sport_games_diff`
- `sport_time_mult_diff`
- overall H2H:
  - `h2h_games`
  - `h2h_avg_diff_p1`
  - `h2h_winrate_p1`
- sport H2H:
  - `sport_h2h_games`
  - `sport_h2h_avg_diff_p1`
  - `sport_h2h_winrate_p1`
- recent features:
  - `sport_diff_mean_10_diff_p1_p2`
  - `sport_resid_mean_10_diff_p1_p2`
  - `sport_diff_std_10_diff_p1_p2`
  - `sport_momentum_diff_5_20_diff_p1_p2`
- long-term features:
  - `sport_long_diff_mean_diff_p1_p2`
  - `sport_long_total_mean_diff_p1_p2`
  - `sport_long_winrate_diff_p1_p2`
  - `sport_p1_long_n`
  - `sport_p2_long_n`

The inclusion of long-term total means makes sense because sport totals are more related to pace and scoring style than differentials alone.

---

## 7.4 Hyperparameters

The final tree system uses three parameter sets.

### 7.4.1 Base differential model parameters

- iterations: `900`
- depth: `5`
- learning rate: `0.03`
- loss function: `RMSE`
- evaluation metric: `RMSE`
- random seed: `12`
- verbose: `False`
- `l2_leaf_reg = 6`

### 7.4.2 Residual differential model parameters

- iterations: `700`
- depth: `5`
- learning rate: `0.03`
- loss function: `RMSE`
- evaluation metric: `RMSE`
- random seed: `12`
- verbose: `False`
- `l2_leaf_reg = 6`

### 7.4.3 Total model parameters

- iterations: `800`
- depth: `6`
- learning rate: `0.03`
- loss function: `MAE`
- evaluation metric: `MAE`
- random seed: `12`
- verbose: `False`
- `l2_leaf_reg = 6`

---

## 7.5 Why these settings make sense

### Tree depth

Depths of 5 and 6 provide enough interaction capacity without allowing overly specific trees.

- depth 5 is fairly conservative
- depth 6 gives a bit more flexibility for totals

### Learning rate

A learning rate of `0.03` is moderate and stable. Lower learning rates often improve generalization but require more trees, which is why iteration counts are moderately large.

### Iteration counts

- base diff gets `900` trees
- residual gets `700`
- total gets `800`

These values reflect the relative complexity of the subtasks.

### Loss choices

- `RMSE` for differential models emphasizes larger errors more strongly
- `MAE` for total model makes the total prediction more robust to occasional outliers

### L2 leaf regularization

`l2_leaf_reg = 6` regularizes leaf values, discouraging excessive correction in small regions of feature space.

---

## 7.6 Sample weighting

The project also used sample weighting to reduce the tendency of the model to become too conservative.

### 7.6.1 Differential weights

For differential models:

$w = 1 + 0.15 \cdot I(|d| \ge 5) + 0.30 \cdot I(|d| \ge 8) + 0.45 \cdot I(|d| \ge 12)$

This gives more importance to stronger wins and losses, which are often harder to predict and easy for a model to underestimate.

### 7.6.2 Residual weights

Residual training uses the same structure, again encouraging attention to larger-magnitude cases.

### 7.6.3 Total weights

For total points:

$w = 1 + 0.02|y_{\text{total}} - 21|$

This lightly emphasizes totals that deviate from a common midrange.

---

## 7.7 Score reconstruction logic

The tree models output continuous predicted values:

- predicted sport differential
- predicted sport total

These are converted into plausible sport scores.

### 7.7.1 Raw reconstruction

The continuous score estimates are:

$s_1 = \frac{1}{2}(T + D)$

$s_2 = \frac{1}{2}(T - D)$

where:

- $T$ = predicted total
- $D$ = predicted differential

### 7.7.2 Legal score decoding for TT/BD/SQ

These sports are decoded into completed game scores of the form:

- `21-x` or `x-21`

The loser score is blended between:

- the reconstructed raw score
- a margin-based loser estimate

This produces realistic outputs rather than mathematically valid but sport-inconsistent scores.

### 7.7.3 Tennis decoding

The system supports two options:

1. independent tennis prediction
2. stop-rule-aware prediction

In practice, the project often used:

- `PREDICT_TENNIS_INDEPENDENTLY = True`

because this was more stable operationally.

---

## 7.8 Training and packaging modes

The model system supports two workflows:

### Evaluation mode

Train on a split and compute metrics.

### Full-data mode

Train on all available rows and save the final model package.

This separation is important because once tuning is finished, the final deployed model should use all data.

---

## 8. Final Inference Package

The saved package includes:

- trained sport models
- differential calibrators
- feature column definitions
- inference state
- metadata

A `PredictorPackage` abstraction exposes methods such as:

- `predict_pair(player1, player2)`
- player state lookups

This makes the model easy to load and use from other scripts or services.

---

## 9. Confidence Experiments Using KD-Trees

A separate line of experiments was developed to connect the project to computational geometry, especially KD-trees.

The motivation was not to replace the main predictive model, but to estimate _confidence_ in predictions using neighborhood geometry.

### 9.1 Core idea

Represent each matchup as a point in feature space. Then use a KD-tree to query nearby historical examples. Confidence might be estimated from:

- neighborhood density
- average neighbor distance
- local consistency of outcomes

---

## 9.2 Experiment 1: Density-only confidence in 4D

Feature space:

- `TT_rating_diff`
- `BD_rating_diff`
- `SQ_rating_diff`
- `TN_rating_diff`

The confidence metric was based on neighborhood closeness or density.

Result:

- it did not behave as expected
- denser regions were not necessarily lower-error regions
- in some settings the relationship was reversed

This showed that simple local density is not automatically a good proxy for predictive reliability.

---

## 9.3 Experiment 2: Local consistency confidence

Instead of density, confidence was based on how similar the outcomes of neighboring points were.

If nearby historical matches had very similar target values, that local region should be more predictable.

This worked better for a local weighted KNN regressor and showed that outcome consistency is more meaningful than density alone.

---

## 9.4 Experiment 3: Applying geometry to the boosted-tree predictor

The same confidence idea was applied to the stronger gradient-boosted tree predictor, still in the simple 4D rating-difference space.

This did not work well. The confidence signal had almost no useful relationship with prediction error.

Interpretation:
the feature space used for geometry was too simple relative to the richer nonlinear predictive model.

---

## 9.5 Experiment 4: Expanded geometric feature space

The KD-tree space was expanded to include not only rating differences but also recent and residual summary features.

This improved the confidence signal slightly, but only modestly.

### Main conclusion from confidence work

Geometry-based confidence can work when:

- the predictor is local and neighborhood-based

It becomes much weaker when:

- the predictive model is much richer than the geometric representation used for confidence

This became an important project-level insight.

---

## 10. Service Layer and API-Oriented Design

The final project was not only a research notebook. It was organized into reusable service functions suitable for a Flask app or local API.

The service layer supports:

- scrape matches
- clean match data
- build training data
- train final model
- load predictor
- predict a matchup
- inspect player state
- rebuild the full pipeline

This design makes the project usable as:

- a local analysis toolkit
- a command-line predictor
- a backend service for a web interface

---

## 11. Strengths of the Project

### 11.1 End-to-end pipeline

The project includes everything from web scraping to deployable inference.

### 11.2 Leakage-safe construction

Features are built using only pre-match information.

### 11.3 Sport-specific structure

The system respects the four-sport structure of racketlon rather than collapsing immediately to one crude label.

### 11.4 Strong tabular modeling

Gradient-boosted trees are a strong fit for engineered sports data.

### 11.5 Rich player-state representation

The combination of ratings, recent form, long-term aggregates, and head-to-head data gives a nuanced player representation.

### 11.6 Computational geometry extension

The KD-tree confidence experiments create a meaningful research component beyond basic supervised learning.

---

## 12. Limitations

### 12.1 Hidden factors

The model does not directly observe:

- injuries
- travel
- fatigue
- tournament context
- psychological matchup effects
- aging curves beyond observed history

### 12.2 Identity limitations

If IDs are not consistently used, player identity depends on name normalization.

### 12.3 Tennis simplification

Independent tennis prediction is easier operationally, but full racketlon stop-rule dynamics may carry extra signal.

### 12.4 Confidence mismatch

Geometric confidence is only weakly aligned with the stronger nonlinear predictor unless the feature space is expanded carefully.

---

## 13. Final Results and Interpretation

The final gradient-boosted tree regression system produced the best practical results among tested methods. It combined:

- dynamic player ratings
- recent-form signals
- long-term shrunk summaries
- matchup history
- nonlinear modeling
- score reconstruction
- save/load inference packaging

It outperformed the simpler baselines both quantitatively and qualitatively.

The confidence experiments also contributed a useful secondary result:

- geometry-based local consistency can be informative
- but confidence estimation is highly dependent on representation choice

---

## 14. Conclusion

This project successfully built a complete sports prediction system for racketlon, beginning from raw scraped tournament data and ending with reusable prediction functions.

The major contributions of the project are:

1. A robust scraper for multiple tournament page formats
2. A strict cleaning and normalization pipeline
3. Leakage-safe pre-match feature generation
4. A nonlinear sport-specific dynamic rating system
5. A final gradient-boosted decision tree regression system for prediction
6. A saved inference package for future matchup prediction
7. Confidence experiments grounded in KD-tree neighborhood geometry

More broadly, the project demonstrates how machine learning, data engineering, and computational geometry ideas can be combined into a full real-world sports analytics workflow.

---

## 15. Exact Final Modeling Details

### 15.1 Rating model constants

- `BASE = 0.0`
- `MAX_DIFF = 21.0`
- TT alpha: `0.060`
- BD alpha: `0.080`
- SQ alpha: `0.060`
- TN alpha: `0.040`

### 15.2 Rating update constants

- `ETA_MIN = 0.04`
- `ETA_MAX = 0.30`
- `ETA_TAU = 20.0`
- `L2 = 1e-5`

### 15.3 Margin multipliers

- TT: `0.6`
- BD: `0.8`
- SQ: `0.6`
- TN: `0.4`

### 15.4 Time multiplier constants

- TT: `c = 0.6`, `tau = 90`
- BD: `c = 0.7`, `tau = 120`
- SQ: `c = 0.6`, `tau = 90`
- TN: `c = 0.5`, `tau = 120`

### 15.5 Long-term shrinkage

- `LONG_TERM_SHRINK = 10.0`

### 15.6 Boosted-tree model parameters

#### Base differential model

- iterations: `900`
- depth: `5`
- learning rate: `0.03`
- loss: `RMSE`
- eval metric: `RMSE`
- random seed: `12`
- `l2_leaf_reg = 6`

#### Residual differential model

- iterations: `700`
- depth: `5`
- learning rate: `0.03`
- loss: `RMSE`
- eval metric: `RMSE`
- random seed: `12`
- `l2_leaf_reg = 6`

#### Total-points model

- iterations: `800`
- depth: `6`
- learning rate: `0.03`
- loss: `MAE`
- eval metric: `MAE`
- random seed: `12`
- `l2_leaf_reg = 6`

### 15.7 Residual combination and calibration

- residual scale: `0.70`
- slope clip: `[0.7, 1.8]`
- intercept clip: `[-5.0, 5.0]`

---

## 16. Reproducibility Summary

The project is reproducible through the following stages:

1. `match_scraper.py`  
   scrape raw tournaments into `matches.csv`

2. `clean_matches.py`  
   filter and normalize into `matches_cleaned.csv`

3. `features.py`  
   build `data.csv` and `inference_state.pkl`

4. model training service  
   train full gradient-boosted tree package on all data

5. API/service functions  
   load saved package and predict future matchups from player names

This structure makes the project suitable both as an academic final project and as a real deployed sports prediction tool.
