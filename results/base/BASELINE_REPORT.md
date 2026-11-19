# Statistical baselines on S1 (driver x day): BASELINE_REPORT

Data: MOCK `data/mock/realdata-v1`

Reproduce: see `src/base/README.md`

## 1. Data protocol (shared with all methods: `src/data/realdata_v1.py`)

| Item | Value |
|---|---|
| Features (order) | n_completed, n_cancelled, n_offered, online_time_h, total_stoppoints, stoppoints_rush (no derived fields) |
| Panel | 3000 drivers x 123 days (2026-07-01 .. 2026-10-31); online_time_h clipped to <= 24 (raw max 26.82) |
| Split | by driver, group-stratified, seed 0: 2100 train / 300 val / 600 test drivers, disjoint (asserted) |
| Window | L = 28 (configurable: `load_s1(window=...)`) |
| Train windows | stride 7 -> **(29400, 28, 6)** |
| Val / test windows | non-overlapping, starts at day 0/28/56/84 (all Wednesdays) -> **(1200, 28, 6)** / **(2400, 28, 6)**; the last 11 days are unused |
| Scaling | per-feature min-max fit on train windows only: min [0,0,0,3,0,0], max [52,9,57,24,72,33]. Metrics run on scaled data; models generate raw units |
| Conditions | group = vehicle_model x fleet (7 groups), day of week, holiday flag (assumed 2026-09-02/03) |
| Generated samples | n = number of real windows of the split (2400 on test), with the test windows' group and start day |

Both methods use all three conditions:
- **group:** one model per vehicle_model x fleet group;
- **day of week:** seasonal terms;
- **holiday:** a shift term.

## 2. Methods (`src/base/`)

- **M2 pooled VAR** (`var_sarima.py`). One VAR(p) per group, pooled OLS over all drivers in the group. It has
  day-of-week means plus a holiday shift, Gaussian innovations, and a driver random effect: each driver gets a
  mean offset and log-scale drawn from a fitted MVN. Covariances are optionally shrunk toward their diagonal.
  Post-processing rounds, clips at >= 0, sets online <= 24, rush <= total and offered >= completed + cancelled.
  - "SARIMA" is realised as a VAR with deterministic weekly dummies plus seasonal AR lag 7 (p = 7). There is no
    MA or differencing part; the 28-day windows are too short for a seasonal MA to be useful.
- **M3 Gaussian copula + AR** (`copula_ar.py`).
  - Marginals per group and feature: empirical, or parametric (negative binomial by moments for counts,
    lognormal for online hours).
  - Count ties use randomized PIT (or midpoint).
  - In normal-score space, a per-feature AR(p) with dow and holiday terms plus a driver mean effect.
  - Cross-feature dependence comes from a Gaussian copula on the innovations, shrinkable toward I.
  - The same post-processing as M2 is applied.

## 3. Search spaces and selection

| Method | Grid | Configs |
|---|---|---|
| M2 | p {1, 2, 7} x seasonal {T, F} x shrink {0, 0.5} x random_effect {off, mean, mean_scale} | 36 |
| M3 | p {1, 7} x family {empirical, parametric} x shrink {0, 0.5} x ties {randomized, midpoint} | 16 |

Every config ran once on **validation** (seed 0), 52 trials in all. **Criterion:** within each method, take the mean
of the 4 metric ranks (lower is better) and break ties by discriminative score. The overall ranking uses the same
criterion over the two selected configs. Test was used only for the final 5-seed runs.

**Best params** (`best_params.yaml`; identical to the selection made before M1 was removed):

| Method | Selected config | Val disc. / pred. / C-FID / corr. | Overall val mean rank |
|---|---|---|---|
| M2 | p 7, seasonal, shrink 0, random_effect mean_scale | 0.2083 / 0.0646 / 0.0892 / 0.0186 | 1.75 |
| **M3 (overall best)** | p 1, parametric (NB / lognormal), shrink 0, randomized ties | 0.1333 / 0.0636 / 0.0509 / 0.0214 | **1.25** |

- M3 wins 3 of the 4 validation metrics (M2 wins correlational).
- **Shrinkage (0.5) is always worse.** It is the clearest effect in the sweeps (figure 7a).
- **M2:** p and the seasonal flag matter little; a random effect clearly beats `off`.

## 4. Final test metrics (best config x 5 seeds, mean ± std, lower is better)

| Method | Discriminative | Predictive (TSTR MAE) | Context-FID | Correlational |
|---|---|---|---|---|
| M2 pooled VAR | 0.2073 ± 0.0238 | 0.0647 ± 0.0002 | 0.0713 ± 0.0035 | 0.0163 ± 0.0047 |
| M3 copula + AR | 0.1969 ± 0.0189 | 0.0639 ± 0.0001 | 0.0313 ± 0.0037 | 0.0278 ± 0.0029 |
| *Real floor: real train windows vs test* | *0.0156 ± 0.0076* | *0.0636 ± 0.0000* | *0.0121 ± 0.0014* | *0.0153 ± 0.0034* |

The real floor uses real train-driver windows with the same start days as the test windows. It is the value a
perfect generator would get under this exact metric code. It is a metric reference, not a method.

**Reading the table:**
- Predictive MAE barely separates methods (0.0636 – 0.0647). Next-day prediction on this data is dominated by
  per-driver noise, so it is a weak metric here.
- On discriminative, both models are far from the floor (≈ 0.2 vs 0.016), and M2 vs M3 is within seed noise.
- M3 is clearly better on Context-FID (0.031 vs 0.071).
- M2 is better on correlational, close to the floor; M3 is about 2x the floor.

## 5. Diagnostics

**Memorization** (seed 0, normalized windows, nearest train window by L2):

| Method | Median nn-train distance | Exact copies (< 1e-6) | Share closer than real-test p5 |
|---|---|---|---|
| M2 | 1.045 | 0% | 4.1% |
| M3 | 1.048 | 0% | 3.1% |
| *Real test windows* | *1.035* | *-* | *5% by definition* |

Both models sit at the same distance from train as unseen real drivers, so neither shows signs of memorization.

**Constraint pass rates** (share of driver-days, seed 0, test conditions):

| Check | Real test | M2 raw | M2 post | M3 raw | M3 post |
|---|---|---|---|---|---|
| counts >= 0 | 1.00 | 0.801 | 1.00 | 1.00 | 1.00 |
| rush <= total stops | 1.00 | 0.987 | 1.00 | 0.995 | 1.00 |
| offered >= completed + cancelled | 1.00 | 0.699 | 1.00 | 0.811 | 1.00 |
| online <= 24 h | 1.00 | 0.9999 | 1.00 | 1.00 | 1.00 |
| completed/online in real 99% band | 0.989 | 0.973 | 0.972 | 0.973 | 0.973 |
| corr(completed, online) | 0.809 | 0.789 | 0.788 | 0.782 | 0.782 |
| mean completed / online h | 1.587 | 1.565 | 1.565 | 1.544 | 1.544 |

- The hard constraints hold only because of post-processing. M2's raw output breaks the offered constraint on 30%
  of days, M3's on 19%.
- The soft "≈ 1.6 trips per online hour" coupling is slightly weakened in both models.

**Runtime and memory** (mean over the 5 test seeds; CPU, 4 processes x 4 threads):

| Method | Fit | Sample 2400 windows | Peak Python memory (tracemalloc) |
|---|---|---|---|
| M2 | 0.39 s | 0.42 s | 63 MB |
| M3 | 12.1 s | 2.0 s | 66 MB |

- Evaluating the 4 metrics costs about 300 s per trial on CPU and dominates everything.
- tracemalloc does not see torch or BLAS native allocations; the memory numbers cover the Python side only.

## 6. Figure gallery (`figures/`, seed-0 samples vs real test; M2 orange, M3 green, real dark grey)

1. **`1_tsne_pca.png`, embeddings.**
   - Both models cover the real support, including the small separate cluster.
   - M2 is denser in the centre and leaves gaps in the lower-right (PCA).
   - M3 spreads too wide (points beyond the real cloud at PCA 1 > 2 and PCA 2 < -0.6) and is less concentrated in
     the dense real cores.
2. **`2_marginals.png`, marginals.**
   - M3 matches the count marginals closely.
   - **M2 smears discrete structure**: too few zero cancellations (0.38 vs 0.46), extra mass at 0 for completed /
     stops / rush from clipping Gaussian draws, and a flattened rush mode.
   - **Both models miss online_time's spiky shape**: real has sharp modes near 8.5-9 h and 11 h (shift-like
     schedules); M2 and M3 produce a smooth bump.
3. **`3_acf.png`, ACF to lag 14 with 95% resampling bands.**
   - The lag-7 peaks are reproduced by both models.
   - **M3's online_time ACF is too high at every lag (≈ 0.80 vs 0.75)**: too much persistence, from an
     over-dispersed driver mean in z-space.
   - M2 slightly over-states n_cancelled autocorrelation (≈ 0.18 vs 0.15). All other curves sit inside or on the
     real band.
4. **`4_crosscorr.png`, lag-0 cross-correlation.**
   - M2's |diff| is at most 0.02.
   - M3 reaches 0.03: it under-states cancelled with everything and online with everything. This explains its
     correlational score.
5. **`5a_trajectories_examples.png`, `5b_trajectories_mean_band.png`.**
   - Means and 10-90% bands match closely for both models.
   - The example paths show the main qualitative failure: **real drivers are heterogeneous in volatility**. Some
     have nearly flat online time (about 9 h every day), while M2 and M3 give every driver similar day-to-day noise.
   - M2's per-driver scale effect captures level differences (e.g. a low 4-9 h driver) but not "flat schedule"
     drivers.
6. **`6_metric_bars.png`.** The table above with the real floor.
7. **`7a_sweeps.png`, `7b_sweep_scatter.png`.** Shrinkage hurts both models; random effects help M2; p and the
   marginal family matter little for M3.
8. **`8_weekly_profile.png`, weekly profile.**
   - Weekday shape is right for both (Tuesday low, Saturday peak; rush peaks on Friday).
   - **M2 inflates n_cancelled by about 10% on every weekday.** This is rounding and clipping bias of Gaussian draws
     on a low-count, zero-heavy feature.
   - Both inflate n_offered by about 0.2-0.4 per day, from the offered >= completed + cancelled floor.
9. **`9_memorization.png`.** M2 and M3 nearest-train distances match the real-test distribution, with 0% exact
   copies.

## 7. Main weaknesses (from the figures)

1. **Discrete, zero-heavy counts under Gaussian dynamics (M2).** Clipping and rounding move mass to 0 and bias
   cancellations up by about 10%. M3's NB marginals fix this.
2. **Per-driver regimes and volatility heterogeneity (M2, M3).** Real online time is spiky (fixed-shift drivers
   with nearly constant hours). Both models produce smooth marginals and homogeneous noise. M3 also overshoots
   online-time persistence.
3. **Hard constraints are imposed, not learned.** Raw model output breaks offered >= completed + cancelled on
   19-30% of days. The post-processing floor then inflates n_offered, and the completed-per-hour coupling is
   slightly weaker (corr 0.78-0.79 vs 0.81).

## 8. Limitations and metric deviations (shared by all methods, documented in `utils/`)

- **Discriminative score:**
  - 2-layer GRU, hidden 16, 2000 iterations, batch 128, 80/20 split, in PyTorch. Yoon et al. use 1 layer with
    hidden = dim/2.
  - Seed noise is ±0.02, so differences under about 0.03 are not meaningful (M2 vs M3 is one of these).
  - The metric is weak on collapsed outputs: constant sequences only reach 0.106 (unit test).
- **Predictive score:** next-step prediction of all 6 features (TimeGAN predicts only the last feature). It barely
  separates methods on this data.
- **Context-FID:** TS2Vec from Diffusion-TS, refit on the real set per evaluation, on CPU.
  - A singular covariance product (a collapsed set) gets pytorch-fid's 1e-6·I offset instead of returning NaN.
  - This fallback never triggered in the logged trials.
- **Correlational score:** Diffusion-TS CrossCorrelationLoss / 10.
- **Selection noise:** selection used a single validation seed. Some configs are within seed noise of each other.
- **Exact-copy tolerance:** 1e-6 in normalized units, because samples are stored as float32.
- **Unused days:** the last 11 days of val/test drivers are not used (4 x 28 = 112 of 123 days).

## 9. Open questions

1. **Holiday dates.** 2026-09-02/03 are assumed (EDA 4.3). Please confirm the real holiday calendar.
2. **Window length.** Is 28 days right for S1, or should it be 7 or 56? Longer windows would leave fewer test
   windows.
3. **offered >= completed + cancelled.** Should this be enforced by post-processing (inflates n_offered) or left
   as a diagnostic only?
4. **Feature list.** online_time is the hardest feature (spiky, shift-like). Keep it as a continuous feature, or
   model schedule type explicitly?
5. **Context-FID encoder.** Should TS2Vec be trained once on train data and frozen for all methods? That is
   cheaper and more comparable than refitting per evaluation.
6. **Real floor.** It is built from real train windows. Is a real reference acceptable in the figures and tables,
   or should it also be dropped?
