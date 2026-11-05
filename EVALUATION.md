# CinePulse Evaluation Report & Offline Benchmark Analysis

This report documents the rigorous offline evaluation protocol, mathematical metric definitions, experimental setup, audit checklist, and empirical findings across the entire CinePulse recommendation pipeline.

---

## 1. Evaluation Methodology

### 1.1 Dataset & Ground Truth Integrity
- **Benchmark Corpus:** GroupLens MovieLens 1M (1,000,209 raw ratings across 3,883 movies and 6,040 users).
  - 598 ratings filtered for movies with fewer than 5 ratings (`min_movie_ratings = 5`).
  - 999,611 valid ratings retained across 3,416 movies and 6,040 users.
- **Split Protocol:** Strict **Leave-Last-2-Out Temporal Splitting**.
  - Train: 987,531 ratings ($t < t_{\text{penultimate}}$)
  - Validation: 6,040 ratings ($t = t_{\text{penultimate}}$)
  - Test: 6,040 ratings ($t = t_{\text{last}}$)
- **Evaluation Cohort:** 500 test users evaluated per cohort with fixed random seed ($S=42$).
- **No Target Leakage:** Candidate scoring, 17-feature extraction, and history sequences are strictly evaluated against the test item using *only* representations derived from prior training interactions. Overlap between training data and validation/test holdouts is verified at 0.00%.

---

## 2. Metric Formulations

### 2.1 Ranking & Retrieval Metrics

#### Recall@K
Measures whether the true held-out test item $i_{\text{test}} \in \mathcal{T}_u$ appears within the top-$K$ recommended candidates $\hat{R}_u$:

$$\text{Recall@}K = \frac{1}{|U|} \sum_{u \in U} \mathbb{I}(i_{\text{test}} \in \hat{R}_{u, 1:K})$$

#### Positive-Only Recall@20 (Strict Endorsement)
To ensure ratings of 1–2 stars are not incorrectly treated as successful recommendations, Positive-Only Recall evaluates strictly against items endorsed with a rating $r \ge 4.0$. 

The denominator is strictly defined as the cohort of users who actually endorsed their held-out item ($|U_{\text{pos}}| = 358$ for the untouched test cohort, $|U_{\text{pos}}| = 344$ for the validation cohort):

$$\text{Recall@20}_{\text{pos}} = \frac{1}{|U_{\text{pos}}|} \sum_{u \in U_{\text{pos}}} \mathbb{I}(i_{\text{test}} \in \hat{R}_{u, 1:20} \land r_{u, i_{\text{test}}} \ge 4.0)$$

#### Normalized Discounted Cumulative Gain (NDCG@K)
Evaluates ranking position quality, penalizing relevant items placed deeper in the recommendation list:

$$\text{DCG@}K = \sum_{r=1}^K \frac{2^{\text{rel}_r} - 1}{\log_2(r + 1)}, \quad \text{NDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K}$$

#### Mathematical Note on Graded vs. Binary NDCG Under Leave-One-Out
Under standard leave-last-1-out evaluation, each test user has exactly 1 held-out test item $m^*$ with rating $r$. 
- When $r \ge 3.0$, the relevance gain is $g = 2^{\text{rel}(r)} - 1 > 0$ ($g=7$ for 5★, $g=3$ for 4★, $g=1$ for 3★).
- If retrieved at rank $k \in \{0, \dots, K-1\}$: $\text{DCG} = \frac{g}{\log_2(k + 2)}$.
- The ideal ranking places the single item at rank 0: $\text{IDCG} = \frac{g}{\log_2(2)} = g$.
- Therefore:
  $$\text{NDCG} = \frac{\text{DCG}}{\text{IDCG}} = \frac{g / \log_2(k + 2)}{g} = \frac{1}{\log_2(k + 2)}$$
The scalar gain $g$ cancels out identically in the numerator and denominator for any $|GT| = 1$. Consequently, graded NDCG and binary NDCG are mathematically identical under leave-last-1-out evaluation.

#### Mean Reciprocal Rank (MRR@K)
Measures the reciprocal rank of the first relevant item:

$$\text{MRR@}K = \frac{1}{|U|} \sum_{u \in U} \frac{1}{\text{Rank}(i_{\text{test}})}$$

---

### 2.2 Beyond-Accuracy Metrics

#### Catalog Coverage@K
Measures the percentage of unique items across the entire catalog $\mathcal{I}$ that the model is capable of recommending across all users:

$$\text{Coverage@}K = \frac{\left| \bigcup_{u \in U} \hat{R}_{u, 1:K} \right|}{|\mathcal{I}|}$$

#### Intra-List Diversity (ILD)
Measures the pairwise genre dissimilarity among the top-$K$ recommended items for user $u$:

$$\text{ILD}(u) = \frac{2}{K(K - 1)} \sum_{i \in \hat{R}_u} \sum_{j \in \hat{R}_u, j > i} \left( 1 - \text{CosineSim}(g_i, g_j) \right)$$

where $g_i$ is the multi-hot genre vector of movie $i$.

#### Novelty (Bits of Self-Information)
Quantifies the unexpectedness or long-tail discovery of recommended items based on inverse item popularity:

$$\text{Novelty}(u) = \frac{1}{K} \sum_{i \in \hat{R}_u} -\log_2 \left( \frac{N_i + 1}{\sum_{j \in \mathcal{I}} N_j} \right)$$

---

## 3. Empirical Results Matrix

### 3.1 Untouched Holdout Cohort Benchmark (Users 500–1000, 95% Bootstrap CIs)

Evaluated on 500 test users from `ratings_test.parquet` never inspected during prior debugging (`artifacts/metrics/evaluation_results_test_untouched.json`):

| Model Architecture | Recall@20 (95% CI) | Recall@100 | NDCG@10 (95% CI) | Recall@20 (Pos $\ge$ 4.0) | Catalog Coverage@20 | Catalog Coverage@100 | Intra-List Diversity | Novelty (bits) | P95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | 0.0360 [0.020, 0.052] | 0.1300 | 0.0112 [0.005, 0.018] | 0.0447 | 0.0261 (2.6%) | 0.1007 (10.1%) | 0.8244 | 10.30 | 0.09 ms |
| **Item-based CF** | 0.1160 [0.088, 0.144] | 0.3160 | 0.0410 [0.027, 0.057] | 0.1341 | 0.1294 (12.9%) | 0.4186 (41.9%) | 0.7161 | 9.53 | 4.00 ms |
| **Content-based Baseline** | 0.0200 [0.010, 0.034] | 0.1140 | 0.0059 [0.002, 0.012] | 0.0168 | 0.3826 (38.3%) | 0.8358 (83.6%) | 0.0641 | 13.03 | 0.29 ms |
| **Two-Tower (ANN Retrieval)** | 0.1060 [0.080, 0.134] | 0.3160 | 0.0346 [0.023, 0.048] | 0.1061 | **0.8188 (81.9%)** | **0.9985 (99.9%)** | 0.4581 | 12.39 | 2.96 ms |
| **Two-Tower + LightGBM Ranker** | **0.1880 [0.154, 0.222]** | **0.4440** | **0.0641 [0.048, 0.082]** | **0.1844** | 0.4148 (41.5%) | 0.7178 (71.8%) | 0.5965 | 10.35 | 15.38 ms |
| **Full Pipeline (Retrieval + Ranker + MMR)** | 0.1740 [0.142, 0.206] | 0.4420 | 0.0601 [0.044, 0.078] | 0.1760 | 0.4081 (40.8%) | 0.7137 (71.4%) | **0.6708** | 10.32 | **20.88 ms** |

---

### 3.2 Validation Dev Cohort Benchmark (Users 0–500, 95% Bootstrap CIs)

Evaluated on 500 validation users from `ratings_val.parquet` (`artifacts/metrics/evaluation_results_val_dev.json`):

| Model Architecture | Recall@20 (95% CI) | Recall@100 | NDCG@10 (95% CI) | Recall@20 (Pos $\ge$ 4.0) | Catalog Coverage@20 | Catalog Coverage@100 | Intra-List Diversity | Novelty (bits) | P95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Popularity Baseline** | 0.0480 [0.030, 0.070] | 0.1440 | 0.0119 [0.005, 0.021] | 0.0640 | 0.0228 (2.3%) | 0.0837 (8.4%) | 0.8257 | 10.27 | 0.09 ms |
| **Item-based CF** | 0.1140 [0.086, 0.144] | 0.3320 | 0.0324 [0.022, 0.044] | 0.1424 | 0.1145 (11.5%) | 0.4019 (40.2%) | 0.7212 | 9.38 | 3.42 ms |
| **Content-based Baseline** | 0.0400 [0.024, 0.058] | 0.1080 | 0.0094 [0.004, 0.017] | 0.0407 | 0.3481 (34.8%) | 0.8094 (80.9%) | 0.0637 | 12.96 | 0.25 ms |
| **Two-Tower (ANN Retrieval)** | 0.1580 [0.126, 0.192] | 0.3580 | 0.0445 [0.030, 0.060] | 0.1541 | **0.8109 (81.1%)** | **0.9974 (99.7%)** | 0.4505 | 12.20 | 2.53 ms |
| **Two-Tower + LightGBM Ranker** | **0.2240 [0.188, 0.264]** | 0.4820 | **0.0659 [0.050, 0.085]** | **0.2297** | 0.4046 (40.5%) | 0.6988 (69.9%) | 0.5836 | 10.21 | 14.17 ms |
| **Full Pipeline (Retrieval + Ranker + MMR)** | 0.2120 [0.178, 0.248] | **0.4840** | 0.0613 [0.045, 0.079] | 0.2267 | 0.3999 (40.0%) | 0.6967 (69.7%) | **0.6559** | 10.18 | **20.80 ms** |

---

### 3.3 Paired Bootstrap Hypothesis Tests (Full Pipeline vs. Item-CF)

User-level paired bootstrap differences ($\Delta_u = M_u(\text{Full Pipeline}) - M_u(\text{Item-CF})$) computed across 1,000 bootstrap resamples:

| Cohort | Metric | Mean Difference ($\Delta$) | 95% Bootstrap CI | Two-Tailed $p$-value | Statistical Conclusion |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Untouched Test Cohort (Users 500–1000)** | **NDCG@10** | **+0.0191** | **[-0.0027, +0.0414]** | $p = 0.0890$ | **NOT statistically significant** at $\alpha=0.05$ |
| **Untouched Test Cohort (Users 500–1000)** | **Recall@20** | **+0.0580** | **[+0.0220, +0.0940]** | $p < 0.001$ | **Statistically significant** improvement |
| **Validation Dev Cohort (Users 0–500)** | **NDCG@10** | **+0.0289** | **[+0.0108, +0.0476]** | $p = 0.0010$ | **Statistically significant** improvement |
| **Validation Dev Cohort (Users 0–500)** | **Recall@20** | **+0.0980** | **[+0.0580, +0.1380]** | $p < 0.001$ | **Statistically significant** improvement |

**Statistical Methodology & Resolution Disclosure:**
1. **Bootstrap Resolution Limit:** With $B=1,000$ resamples, the finite discrete resolution is $\Delta p = 1/B = 0.001$. An earlier claim of $p < 0.0001$ was an artifact of displaying 4 decimal places ($0.0000$) on 0 extreme resamples. Applying the standard finite-sample adjustment (Davison & Hinkley, 1997), $p = (r + 1)/(B + 1) = 1/1001 \approx 0.0010$, correctly bounded as $p < 0.001$.
2. **Honest Uncertainty Around Test NDCG@10:** While the Full Pipeline point estimate reaches $0.0601$ vs Item-CF's $0.0410$ ($\Delta = +0.0191$), the paired bootstrap 95% confidence interval spans zero ($[-0.0027, +0.0414]$) with $p = 0.0890 > 0.05$. Therefore, we **do not claim** statistical significance on test NDCG@10 at the 95% confidence level. However, Recall@20 shows a clear and statistically significant advantage ($+0.0580$, $p < 0.001$).

---

### 3.4 Multi-Positive Graded Evaluation Audit Finding (EXCLUDED FROM VALID BENCHMARKS)

> [!WARNING]
> **Status: INVALID / METHODOLOGICALLY COMPROMISED (EXCLUDED FROM PERFORMANCE CLAIMS)**
> In `pipelines/evaluate.py:run_multi_positive_evaluation`, holding out the last 5 interactions for evaluation without retraining model artifacts meant that **15,858 of 26,430 test pairs (60.0%)** had already been observed during training in `ratings_train.parquet` (which only held out 2 interactions).
> These metrics are **excluded from all performance claims** and marked invalid across the repository.

**Compromised Metric Records (`artifacts/metrics/multi_positive_graded_results.json`):**
- Item-based CF: Binary NDCG@10 = 0.0228, Graded NDCG@10 = 0.0240, Recall@20 = 0.0527
- Two-Tower + LightGBM Ranker: Binary NDCG@10 = 0.0502, Graded NDCG@10 = 0.0500, Recall@20 = 0.0987
- Full Pipeline: Binary NDCG@10 = 0.0446, Graded NDCG@10 = 0.0466, Recall@20 = 0.0935

**Standalone Procedure to Create a Valid Multi-Positive Benchmark (Requires Full Retraining):**
1. *Data Split:* Regenerate `ratings_train.parquet`, `ratings_val.parquet`, and `ratings_test.parquet` where the last 5 interactions per user ($t_{n-4} \dots t_n$) are held out *prior* to training.
2. *Retrain Retrieval:* Re-train Two-Tower model on the clean $K=5$ train split and rebuild the FAISS candidate index.
3. *Retrain Ranker:* Re-extract candidate feature pools from the $K=5$ retrieval models and fit LightGBM LambdaRank.
4. *Evaluation:* Execute multi-positive graded evaluation against the uncontaminated 5-item test set.

---

### 3.5 MMR Validation Pareto Frontier Study (VAL Split, 500 Validation Users)

Conducted on `ratings_val.parquet` to map the exact trade-off between listwise ranking relevance, diversity, and catalog coverage (`artifacts/metrics/pareto_study_val.json`):

| Configuration | Recall@20 | Recall@100 | NDCG@10 | MRR@10 | ILD | Novelty | Coverage@20 | Latency P95 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MMR Disabled (Pure Ranker)** | 0.2240 | 0.4820 | 0.0659 | 0.0479 | 0.5836 | 10.21 | 0.4046 | 0.05 ms |
| **$\lambda=1.00, \beta=0.00$ (Identity MMR)** | 0.2240 | 0.4820 | 0.0659 | 0.0479 | 0.5836 | 10.21 | 0.4046 | 2.32 ms |
| **$\lambda=1.00, \beta=0.03$ (Relevance + Low Nov)** | 0.2240 | 0.4820 | 0.0635 | 0.0460 | 0.5813 | 10.24 | 0.4078 | 2.40 ms |
| **$\lambda=0.95, \beta=0.00$ (Mild Div, Zero Nov)** | 0.2160 | 0.4860 | **0.0667** | **0.0489** | 0.6090 | 10.18 | 0.4011 | 1.65 ms |
| **$\lambda=0.90, \beta=0.01$ (Conservative)** | 0.2100 | 0.4840 | 0.0620 | 0.0457 | 0.6333 | 10.18 | 0.3990 | 2.13 ms |
| **$\lambda=0.85, \beta=0.03$ (Balanced Production)** | 0.2120 | 0.4840 | 0.0613 | 0.0444 | 0.6559 | 10.18 | 0.3999 | 1.73 ms |
| **$\lambda=0.75, \beta=0.05$ (Exploration)** | 0.2000 | 0.4820 | 0.0583 | 0.0425 | 0.7032 | 10.19 | 0.3972 | 1.76 ms |
| **$\lambda=0.65, \beta=0.15$ (Aggressive / Diversity)** | 0.1660 | 0.4820 | 0.0402 | 0.0276 | **0.7572** | **10.36** | **0.4128** | 1.85 ms |

---

## 4. CinePulse V2 Independent Test Confirmation

To independently validate CinePulse V2 (Configuration 8) free from any confirmation bias or validation split overfitting, a single-pass confirmation evaluation was executed across the held-out temporal test set (`ratings_test.parquet`, zero target leakage). Pre-registered user cohorts and exact reproducible results are preserved in `artifacts/experiments/relevance_v2/test_cohort_manifest.json` and `artifacts/experiments/relevance_v2/test_confirmation_results.json`.

### 4.1 Cohort Architecture & Definitions
1. **Sequestered Cohort (3,974 users):** The primary independent confirmation benchmark (`all_eligible[1000:]`). These users were strictly quarantined and never inspected or evaluated during any exploratory data analysis, v1 debugging, or tournament sweeps.
2. **Untouched Historical Cohort (500 users):** The exact cohort (`all_eligible[500:1000]`) used in historical CinePulse V1 release benchmarking, enabling direct longitudinal comparison.
3. **Full Test Cohort (4,974 users):** All MovieLens users meeting the evaluation eligibility criteria ($r_{\text{test}} \ge 3.0$).
*Note on User Type:* All evaluated test users are **warm MovieLens users** whose historical training interactions ($t < t_{\text{penultimate}}$) are encoded in user representations, with their single final interaction ($t = t_{\text{last}}$) held out.

---

### 4.2 Primary Confirmation Benchmark: 3,974 Sequestered Test Users

| Recommendation Architecture | Recall@20 | Recall@100 | NDCG@10 | Pos NDCG@10 ($r \ge 4.0$) | MRR@10 | Catalog Coverage@100 | Intra-List Diversity | P95 Latency | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CinePulse V1 Production Baseline**<br>*(17f, TT=100, CF=100, Pop=30, MMR 0.85/0.03)* | 18.09% | 40.59% | 0.0560 | 0.0614 | 0.0400 | 72.57% | **0.6933** | **24.86 ms** | **14.19 ms** |
| **CinePulse V2 Pure Ranker**<br>*(24f, TT=200, CF=100, Pop=30, No MMR)* | 22.14% | 44.19% | 0.0768 | 0.0834 | 0.0558 | **95.49%** | 0.4922 | 27.93 ms | 17.17 ms |
| **CinePulse V2 Production Winner (Config 8)**<br>*(24f, TT=200, CF=100, Pop=30, MMR 0.95/0.01)* | **22.19%**<br>*(+22.7%)* | **44.21%**<br>*(+8.9%)* | **0.0778**<br>*(+38.9%)* | **0.0850**<br>*(+38.4%)* | **0.0563**<br>*(+40.7%)* | 95.08%<br>*(+22.5 pp)* | 0.5165 | 27.92 ms | 17.16 ms |

#### Paired Bootstrap Significance ($B=1,000$ Resamples, 3,974 Users)
- **NDCG@10 Difference ($\Delta = \text{V2} - \text{V1}$):** Mean $\Delta = +0.0218$, 95% Bootstrap CI: $[+0.0165, +0.0277]$, two-sided $p = 0.0010$ (significant at $\alpha=0.01$).
- **Recall@20 Difference ($\Delta = \text{V2} - \text{V1}$):** Mean $\Delta = +0.0410$, 95% Bootstrap CI: $[+0.0302, +0.0516]$, two-sided $p = 0.0010$ (significant at $\alpha=0.01$).
- **V2 Config 8 vs. V2 Pure Ranker:** $\Delta \text{NDCG@10} = +0.0010$, 95% CI: $[-0.0009, +0.0029]$, $p = 0.2867$ (no statistically significant loss of relevance, with meaningful genre redundancy reduction).

---

### 4.3 Longitudinal Tracking on Historical Untouched Cohort (500 Users)

| Recommendation Architecture | Recall@20 (95% CI) | Recall@100 | NDCG@10 (95% CI) | Pos NDCG@10 | Catalog Coverage@100 | Intra-List Diversity | P95 Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CinePulse V1 Baseline (Unified Re-evaluation)** | 17.80% [0.144, 0.212] | 43.40% | 0.0530 [0.038, 0.070] | 0.0545 | 64.64% | **0.6867** | **24.86 ms** |
| **CinePulse V1 Baseline (Original Offline Report)** | 17.40% [0.142, 0.206] | 44.20% | 0.0601 [0.044, 0.078] | 0.0612 | 71.37% | **0.6708** | **20.88 ms** |
| **CinePulse V2 Pure Ranker** | 22.00% [0.184, 0.258] | 45.40% | 0.0816 [0.062, 0.102] | 0.0837 | 81.24% | 0.4848 | 27.93 ms |
| **CinePulse V2 Production (Config 8)** | **21.40% [0.178, 0.252]** | **45.40%** | **0.0801 [0.061, 0.100]** | **0.0857** | **81.32%** | 0.5114 | 27.92 ms |

*Paired Bootstrap Significance on 500 Untouched Cohort:*
- $\Delta \text{NDCG@10} = +0.0272$, 95% CI: $[+0.0096, +0.0464]$, $p = 0.0090$.
- $\Delta \text{Recall@20} = +0.0360$, 95% CI: $[+0.0060, +0.0660]$, $p = 0.0180$.

#### Technical Reconciliation: V1 Benchmark Consistency (0.0601 vs. 0.0530)
Both evaluations evaluated the exact same 500 test users (`all_eligible[500:1000]`, MD5 checksum `1012e3ef31110d3b721b324292c8fcbf`). The slight difference between the original offline report (NDCG@10 = 0.0601 in `artifacts/metrics/evaluation_results_test_untouched.json`) and the unified tournament harness (NDCG@10 = 0.0530 in `artifacts/experiments/relevance_v2/test_confirmation_results.json`) stems from two verified implementation factors:
1. **Candidate Retrieval Parameterization:** The original V1 evaluation (`pipelines/evaluate.py`) utilized un-shrunk Item-CF ($K=100, \lambda=0.0$), whereas the unified tournament harness (`pipelines/run_test_confirmation.py`) evaluated all configurations with shrinkage parameter $\lambda=25.0$ and $K=200$ neighbors (`EnhancedItemCF(top_k_neighbors=200, shrinkage_lambda=25.0)`). While shrinkage improved standalone Item-CF precision in isolation, candidate pool selection for the 17-feature ranker differed slightly on specific edge items.
2. **Feature Extractor Vectorization & Calibration:** The original evaluation used `RankingFeatureExtractor` (`src/ranking/features.py`), constructing a Pandas DataFrame row-by-row and applying mean imputation for missing values. The unified harness used `FastRankingFeatureExtractor` (`src/ranking/features_fast.py`), an optimized NumPy matrix extractor that computes calibrated personal rating offsets and vectorized operations directly.
3. **Statistical Compatibility:** Both metrics fall within each other's 95% bootstrap confidence intervals: $0.0530 \in [0.044, 0.078]$ and $0.0601 \in [0.038, 0.070]$. When expanded to the full sequestered cohort of 3,974 test users, V1 NDCG@10 converges to 0.0560 $[0.052, 0.060]$, demonstrating that the 500-user sample naturally exhibits minor sample variance ($\pm 0.004$). Under either baseline, CinePulse V2 delivers a large, statistically significant improvement (+33.3% to +51.1%).

---

### 4.4 Latency & Throughput Benchmark Comparison

Benchmarked on CUDA (NVIDIA GeForce RTX 3050 Laptop GPU, AMD Ryzen 7 5800H):

| Metric | CinePulse V1 (17 Features, TT=100) | CinePulse V2 (24 Features, TT=200) | Production SLA |
| :--- | :---: | :---: | :---: |
| **Candidate Retrieval (ANN + Item-CF + Pop)** | 2.96 ms | 3.10 ms | < 5.0 ms |
| **Feature Extraction & LightGBM Scoring** | 12.27 ms | 14.80 ms | < 25.0 ms |
| **Vectorized Incremental MMR Re-Ranking** | 0.70 ms | 0.70 ms | < 2.0 ms |
| **Mean End-to-End Latency** | 14.19 ms | 17.16 ms | < 20.0 ms |
| **P50 Latency** | 13.80 ms | 21.42 ms | < 25.0 ms |
| **P90 Latency** | 19.50 ms | 27.85 ms | < 30.0 ms |
| **P95 Latency** | **24.86 ms** | **27.92 ms** (32.99 ms peak cold) | **< 35.0 ms** |
| **Inference Throughput** | ~52 QPS | ~43 QPS | > 30 QPS |

---

### 4.5 Production Serving Engine vs. Interactive Static Showcase
To ensure technical transparency, CinePulse distinguishes between its two deployment targets:
- **FastAPI Real-Time Inference Engine (`src/api/main.py`):** The primary production service executing live GPU/CPU neural retrieval via PyTorch Two-Tower, sub-3ms exact inner product candidate retrieval with FAISS, real-time 24-feature extraction, LightGBM LambdaRank scoring, MMR re-ranking, and dynamic session preference vector updating ($\gamma=0.25$) with SQLite persistence.
- **GitHub Pages Showcase (`https://ahirerohit.github.io/CinePulse/`):** A zero-cost, static web application designed for interactive portfolio demonstration. It serves precomputed V2 production candidate recommendations (`showcase_data.json`) and verified evaluation benchmarks (`metrics.json`), while handling real-time user actions (likes, dislikes, instant item suppression, genre filtering, and cold-start onboarding) client-side in React 19.

---

## 5. Deep Empirical Findings & Technical Insights

### 5.1 Why LightGBM Alone (0.0641) Outperforms Full Pipeline (0.0601) on NDCG@10
1. **Listwise Optimization Objective:** The LightGBM ranker directly optimizes the LambdaRank listwise objective, sorting candidates strictly to maximize NDCG.
2. **Intentional Relevance Sacrifice:** MMR intentionally perturbs pure relevance ordering to prevent genre fatigue and redundancy. Moving from pure ranker ($\lambda=1.0$) to balanced MMR ($\lambda=0.85, \beta=0.03$) trades 6.2% NDCG@10 (0.0641 to 0.0601) in exchange for a **+12.5% increase in Intra-List Diversity (0.6708 vs 0.5965)**.
3. **Mild Diversity Alternative:** If an application strictly prioritizes raw ranking metrics above list diversity, the validation Pareto sweep demonstrates that $\lambda=0.95, \beta=0.00$ maintains NDCG@10 = 0.0667 while still offering a mild +4.3% diversity boost over pure ranking.

### 5.2 Reconciling Catalog Coverage (Old 97.2% vs. Current 71.37%)
1. **Root Cause of Old 97.2% Coverage under Production Parameters ($\lambda=0.85, \beta=0.03$):**
   - In the pre-audit pipeline, the Two-Tower model suffered from severe representation collapse (embeddings compressed into a positive hyper-cone with mean similarity 0.533), and the LightGBM ranker was trained on an arithmetic mean proxy ($\mathbf{u} = \frac{1}{|H|}\sum \mathbf{v}_i$) with only 0.29 cosine similarity to serving.
   - Because the pre-audit ranker lacked discriminative precision, it distributed high ranks nearly uniformly across the broad, uncurated Two-Tower candidate pool (Two-Tower ANN alone covered 99.65%). Across 500 users, this noisy dispersion resulted in an artificially high 97.2% union coverage at the expense of relevance (NDCG@10 = 0.0257).
2. **Current 71.37% Production Coverage:**
   - With LayerNorm, temperature $\tau=0.05$, anti-collapse normalization, and exact user tower vector training for LightGBM, the ranker learns sharp preference boundaries, concentrating top ranks on high-relevance movies fitting users' genuine tastes.
   - The resulting 71.37% catalog coverage represents a healthy **+70.5% wider catalog exploration than Item-CF (41.86%)** while lifting NDCG@10 to 0.0601. Furthermore, standalone Two-Tower ANN retrieval covers **99.85%** of the catalog, confirming that candidate generation leaves no part of the catalog stranded.
3. **Normalization Architecture:**
   - The MMR implementation in `src/reranking/mmr.py` utilizes **standardized sigmoid normalization**:
     $$z_i = \frac{s_i - \mu_s}{\max(\sigma_s, 0.25)}, \quad \text{norm\_scores}_i = \frac{1}{1 + e^{-z_i}}$$
   - Legacy min-max scaling ($s' = \frac{s - \min}{\max - \min}$) was completely eliminated because it artificially forced the #1 ranked candidate to always evaluate to 1.0.

### 5.3 Feature Parity & Evolution from 17 to 24 Features
The LightGBM ranker extracts features with verified exact parity between training and inference. In V2, 7 causal interaction and calibration features were added to the baseline 17 features:
1. `candidate_rank`: 0-indexed candidate position in retrieval pool
2. `two_tower_score`: Genuine inner-product similarity $\langle e_i, u \rangle$
3. `cf_score`: Collaborative filtering co-occurrence score
4. `genre_overlap`: Dot product between user history genre profile and movie genres
5. `genre_cosine`: Cosine similarity between user genre profile and movie genres
6. `log_rating_count`: $\log(1 + N_i)$
7. `rating_mean`: Raw arithmetic average rating
8. `rating_std`: Rating standard deviation
9. `bayesian_rating`: Bayesian smoothed rating prior
10. `release_year`: Item release year
11. `year_diff`: Absolute difference between item release year and user mean year
12. `user_activity_count`: Number of past user ratings
13. `is_source_two_tower`: Indicator for Two-Tower retrieval
14. `is_source_item_cf`: Indicator for Item-CF retrieval
15. `is_source_popularity`: Indicator for Popularity retrieval
16. `cf_x_bayesian`: Interaction term between CF score and centered Bayesian rating
17. `tt_x_genre`: Interaction term between Two-Tower score and genre cosine
18. `cf_score_normalized`: Pool-normalized collaborative filtering score
19. `tt_score_normalized`: Pool-normalized Two-Tower dot product
20. `score_product`: Consensus interaction (`cf_score_normalized` $\times$ `tt_score_normalized`)
21. `score_ratio`: Dynamic consensus ratio (`tt_score_normalized` / (`cf_score_normalized` + $10^{-4}$))
22. `user_rating_calib_diff`: User rating calibration delta (`bayesian_rating` - `user_mean_rating`)
23. `activity_popularity_interaction`: User engagement $\times$ item popularity interaction
24. `release_year_decay`: Exponential decay from user mean release year

### 5.4 End-to-End Latency Verification
Benchmarked with vectorized 24-feature extraction and incremental MMR:
- Two-Tower FAISS Exact Retrieval (Top 200): ~3.10 ms
- Shrunk Item-CF Retrieval (Top 100): ~0.18 ms
- 24-Feature Extraction & LightGBM Scoring: ~14.80 ms
- Vectorized Incremental MMR Re-Ranking: ~0.70 ms
- Database & Session State Overhead: ~1.50 ms
- **Total Measured P95 Latency: 27.92 ms** (well below the 35 ms production SLA)
