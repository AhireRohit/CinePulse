# CinePulse: Release Candidate Integration & Final Validation Report

**Candidate Version:** Configuration 8 (`model_version = "v2"`)  
**Status:** **READY TO DEPLOY** (Awaiting User Authorization)  
**Branch:** `experiment/relevance-v2` (Local Only — Zero Remote Commits/Pushes)  
**Evaluation Scope:** 3,974 Sequestered Test Users & 4,974 Full Test Users (`ratings_test.parquet`)  
**Test Suite Status:** 43/43 Tests Passing (Unit, Integration, E2E, Safeguards, Serving Parity)  

---

## 1. Executive Summary & Verdict

Following the completion of the 6-experiment ML relevance tournament and the rigorous validation of finalist architectures against the corrected published production baseline, **Configuration 8** was selected as the new candidate release. 

Over this phase, Configuration 8 has been **fully integrated into the local CinePulse serving engine** (`src/pipeline/engine.py`, `src/ranking/feature_engineering.py`, `src/models/ranker.py`, `src/api/main.py`) with complete backward compatibility via a configurable toggle (`model_version = "v2"` vs `"v1"`).

### Core Validation Highlights

1. **Defensible Independent Test Confirmation:**
   - Evaluated on **3,974 strictly sequestered test users** in `ratings_test.parquet` (users never inspected during prior development or debugging).
   - Achieved **Recall@20 = 0.2219** (+22.7% relative gain over published baseline `0.1809`).
   - Achieved **NDCG@10 = 0.0778** (+38.9% relative gain over published baseline `0.0560`).
   - Achieved **Catalog Coverage@100 = 95.1%** (+22.5 percentage points over published baseline `72.6%`).
   - Paired bootstrap test ($B=1,000$ resamples) proves statistical significance: $\Delta \text{NDCG@10} = +0.0218$, 95% CI $[+0.0165, +0.0277]$, empirical $p = 0.0010$.
2. **Mathematical Serving Parity Verified:**
   - Automated parity tests across multiple real MovieLens users confirm that live in-process serving (`RecommendationEngine`) and experimental evaluation produce identical candidate sets, identical 24-feature vectors (error $< 10^{-4}$), identical booster scores (error $< 10^{-5}$), and identical final top-20 item IDs.
3. **Product Safeguards & Regression Free:**
   - Demo personas (The Cinephile #42, Sci-Fi Explorer #100, Classic Drama #250, Indie Animation #500), sparse/dense profiles, and cold-start onboarding operate reliably.
   - Quality-Aware Hero Spotlight guarantees that movies featured in the hero card satisfy strict acclaim criteria ($\ge 3.20$ Bayesian rating, $\ge 3.10$ mean, $\ge 50$ ratings) while allowing personalized niche exploration in the regular recommendation feed.
   - 43/43 test suite passes; frontend build and ruff linting are 100% clean; local serving latency is 21.42 ms (P50) and 32.99 ms (P95) at 43.2 QPS.

---

## 2. Architecture & Configuration Specification

Configuration 8 optimizes the entire multi-stage pipeline:

```
[User Request / Profile]
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      STAGE 1: EXPANDED CANDIDATE RETRIEVAL                       │
│  • Two-Tower ANN Retrieval (FAISS IndexFlatIP, 64-dim): Top 200 items           │
│  • Shrunk Item-CF (Vectorized Co-occurrence, K=200, λ=25.0): Top 100 items      │
│  • Popularity Discovery (Cold-start fallback & trending): Top 30 items          │
│  Union Pool: ~280–315 unique candidates per request                             │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                   STAGE 2: 24 CAUSAL FEATURE EXTRACTION                         │
│  Baseline (17 Features):                                                        │
│    candidate_rank, two_tower_score, cf_score, genre_overlap, genre_cosine,       │
│    log_rating_count, rating_mean, rating_std, bayesian_rating, release_year,    │
│    year_diff, user_activity_count, is_source_two_tower, is_source_item_cf,      │
│    is_source_popularity, cf_x_bayesian, tt_x_genre                              │
│  7 New High-Fidelity Causal Features:                                           │
│    source_count, cf_score_normalized, tt_score_normalized, score_product,        │
│    user_rating_calib_diff, rating_count_confidence, recent_genre_cosine          │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                  STAGE 3: PURE LIGHTGBM LAMBDARANK SCORING                      │
│  Model Checkpoint: artifacts/experiments/relevance_v2/models/ranker_lgbm_exp3.txt│
│  • 150 Trees, max_depth=6, num_leaves=31, learning_rate=0.05                   │
│  • Exact LambdaRank NDCG objective optimization                                 │
│  • External Score Fusion: DISABLED (Eliminates heuristic distortion)            │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                STAGE 4: RELEVANCE-FIRST MMR DIVERSITY RERANKING                 │
│  Parameters: λ = 0.95 (Relevance Weight), β = 0.01 (Novelty Weight)             │
│  • Prevents diversity penalties from displacing high-relevance candidates       │
│  • Seen-item suppression & dynamic dislike exclusion enforced                   │
└──────────────────────────────────────┬──────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│              STAGE 5: QUALITY-AWARE HERO SPOTLIGHT GATE                         │
│  Enforces Acclaim Floor: bayesian_rating >= 3.20, rating_mean >= 3.10, count >= 50│
│  Picks top candidate with highest merit: relevance * (bayesian / 5.0) * conf    │
│  Fallback: Tasteful platform curation if no personalized candidate meets gate   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Dynamic Backward Compatibility Toggle

In `src/core/config.py` and `src/pipeline/engine.py`, the system supports runtime switching between configurations:
```python
# Toggle in src/core/config.py
model_version: str = "v2"  # "v1" for original published production, "v2" for Config 8
```

---

## 3. Independent Confirmation on Sequestered Test Cohort

### Single-Pass Test Benchmark Results

| Cohort | Configuration | Recall@20 | NDCG@10 | Pos NDCG@10 ($r \ge 4.0$) | Coverage@100 | P95 Latency |
|---|---|---:|---:|---:|---:|---:|
| **Sequestered (3,974 users)** | Published Production Baseline | 0.1809 | 0.0560 | 0.0614 | 72.6% | 24.86 ms |
| | **Promoted Config 8** | **0.2219** | **0.0778** | **0.0850** | **95.1%** | 27.92 ms |
| | New 24f Pure Ranker (No MMR) | 0.2214 | 0.0768 | 0.0834 | 95.5% | 27.93 ms |
| | *Relative Gain (Config 8 vs Baseline)* | *+22.7%* | *+38.9%* | *+38.4%* | *+22.5 pp* | *+3.06 ms* |
| **All Test (4,974 users)** | Published Production Baseline | 0.1811 | 0.0558 | 0.0604 | 72.8% | 24.86 ms |
| | **Promoted Config 8** | **0.2240** | **0.0781** | **0.0845** | **96.0%** | 27.92 ms |
| | New 24f Pure Ranker (No MMR) | 0.2242 | 0.0776 | 0.0833 | 96.3% | 27.93 ms |
| | *Relative Gain (Config 8 vs Baseline)* | *+23.7%* | *+40.0%* | *+39.9%* | *+23.2 pp* | *+3.06 ms* |
| **Untouched (500 users)** | Published Production Baseline | 0.1780 | 0.0530 | 0.0545 | 64.6% | 24.86 ms |
| | **Promoted Config 8** | **0.2140** | **0.0801** | **0.0857** | **81.3%** | 27.92 ms |
| | *Relative Gain (Config 8 vs Baseline)* | *+20.2%* | *+51.1%* | *+57.2%* | *+16.7 pp* | *+3.06 ms* |

### Statistical Significance (Paired Bootstrap, $B=1,000$)

- **Sequestered Cohort (3,974 users):**
  - $\Delta \text{NDCG@10} = +0.0218$, 95% CI: $[+0.0165, +0.0277]$, empirical $p = 0.0010$.
  - $\Delta \text{Recall@20} = +0.0410$, 95% CI: $[+0.0302, +0.0516]$, empirical $p = 0.0010$.
- **All Test Users (4,974 users):**
  - $\Delta \text{NDCG@10} = +0.0224$, 95% CI: $[+0.0173, +0.0271]$, empirical $p = 0.0010$.
  - $\Delta \text{Recall@20} = +0.0428$, 95% CI: $[+0.0330, +0.0525]$, empirical $p = 0.0010$.

---

## 4. Serving Parity Verification

In `tests/test_serving_parity.py`:
- 8/8 tests pass.
- Verified candidate set parity, 24-feature matrix parity ($<10^{-4}$), raw model scores ($<10^{-5}$), and top-20 item IDs across users #10, #42, #100, #250, #500, #1000.
- Reproducibility confirmed across repeated queries.

---

## 5. Product Safeguards & Performance

- **Safeguards (5/5 tests passing):** Demo personas, sparse/dense users, cold-start onboarding, hero spotlight acclaim floor, and real-time dislike suppression.
- **Latency Benchmark (100 calls on CUDA):**
  - Mean: 23.14 ms | P50: 21.42 ms | P95: 32.99 ms | P99: 41.98 ms | Throughput: 43.21 QPS
- **Full Pytest Suite:** 43/43 tests passing in 52.85s.
- **Frontend & Code Quality:** TypeScript build clean in 1.80s, Ruff lint clean.

---

## 6. Release Verdict

**Decision: `READY TO DEPLOY`**  
All validation criteria are met. Awaiting explicit user confirmation before merging `experiment/relevance-v2` into `main` and deploying.
