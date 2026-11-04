# CinePulse Architecture & Technical Specification

This document details the engineering principles, mathematical formulations, algorithmic implementations, and system tradeoffs behind CinePulse.

---

## 1. System Overview & The Industrial Multi-Stage Funnel

Real-world recommendation systems operating at scale (e.g., Netflix, YouTube, Amazon, Meta) cannot evaluate full ranking models over millions of catalog items per request. Instead, they operate as a progressive funnel:

```
+-----------------------------------------------------------+
| Catalog: 3,416 Movies (ML-1M; or 87.5k items in ML-32M)   |
+-----------------------------------------------------------+
                             |
                             v
           [ Stage A & B: Candidate Retrieval ]
       • Two-Tower Deep Neural Network + FAISS ANN
       • Item-Based Collaborative Filtering (Shrunk Cosine)
       • Bayesian Trending / Popularity
       -> Latency: ~1-3 ms | Pool: ~170 Deduped Candidates
                             |
                             v
              [ Stage C: Personalized Ranking ]
       • LightGBM LambdaRank (Listwise NDCG Optimization)
       • 17-Dimensional Cross-Feature Feature Engineering
       • Non-linear user-item affinity & temporal decay
       -> Latency: ~10-15 ms | Pool: Scored Candidates
                             |
                             v
        [ Stage D: Diversity & Novelty Re-Ranking ]
       • Vectorized Maximal Marginal Relevance (MMR)
       • Inverse Log-Frequency Novelty Weighting
       • Dynamic Diversity Slider (λ=0.85, β=0.03 default)
       -> Latency: ~0.7 ms | Pool: Top 15-20 Slotted Items
                             |
                             v
           [ Stage E: Real-Time Session Adaptation ]
       • Instant Negative Feedback Suppression (Dislikes/Dismissals)
       • Dynamic In-Memory User Vector Blending
       • Sub-1ms Recalibration without Neural Retraining
```

---

## 2. Dataset Engineering & Temporal Split Integrity

### 2.1 The Leave-Last-2-Out Temporal Splitting Strategy
A pervasive failure mode in recommender system evaluations is **random train/test splitting**. In temporal domains like movies and streaming:
- Random splitting leaks future items into past training.
- Models memorize specific item combinations rather than predicting temporal user trajectories.

**CinePulse enforces strict temporal splitting:**
1. For each user $u \in U$ with interaction sequence sorted chronologically by timestamp $t$:
   - The final interaction $(u, i_{\text{last}}, t_{\text{last}})$ is reserved exclusively for the **Held-Out Test Set**.
   - The second-to-last interaction $(u, i_{\text{penultimate}}, t_{\text{penultimate}})$ is reserved exclusively for the **Validation Set**.
   - All earlier interactions $\{ (u, i, t) \mid t < t_{\text{penultimate}} \}$ constitute the **Training Set**.
2. **Cold-start & leakage prevention:**
   - Any movie appearing only in the validation or test splits is filtered to prevent target leakage.
   - User historical sequences constructed for embedding generation and ranking features utilize **only** training set interactions.

```
User Timeline:
[ t_0, t_1, t_2, ..., t_{n-3}, t_{n-2} ] -> TRAIN (987,531 ratings)
[ t_{n-1} ]                              -> VALIDATION (6,040 ratings)
[ t_n ]                                  -> TEST (6,040 ratings)
```

### 2.2 Bayesian Smoothed Popularity & Continuous Mappings
Raw arithmetic means ($\bar{r} = \frac{1}{N} \sum r_i$) are noisy for items with very few ratings (e.g., a movie with a single 5.0 rating falsely outranks a classic with 2,000 ratings averaging 4.5).
CinePulse computes Bayesian smoothed ratings:

$$r_{\text{Bayes}} = \frac{C \cdot m + \sum_{j=1}^{N_i} r_{i,j}}{C + N_i}$$

Where:
- $m$: Global mean rating across the training corpus ($m \approx 3.58$).
- $C$: Prior weight confidence constant ($C = 25$).
- $N_i$: Number of observed ratings for item $i$.

---

## 3. Stage B: Two-Tower Neural Candidate Retrieval

### 3.1 Network Architecture
The candidate retrieval model maps high-dimensional user interaction histories and item metadata into a shared $D$-dimensional metric space ($D=64$), where inner product corresponds to recommendation affinity.

#### User Tower $f_{\theta}(u, \mathcal{H}_u, g_u)$:
1. **User Identity Embedding:** $e_u \in \mathbb{R}^D$ (supports existing users).
2. **Sequential History Representation:**
   - For user history $\mathcal{H}_u = [i_1, i_2, \dots, i_L]$ (truncated to $L=20$ most recent items):
   - Sequence items are embedded: $E_H \in \mathbb{R}^{L \times D}$.
   - **Additive Attention Sequence Pooling:** Rather than naive average pooling, an attention weight vector $\alpha \in \mathbb{R}^L$ is computed dynamically via a learned projection layer $W_{\text{att}} \in \mathbb{R}^{D \times 1}$:
     $$\alpha = \text{Softmax}\left( E_H W_{\text{att}} + b_{\text{att}} + M \right)$$
     where $M$ is a padding mask ($M_k = -10^4$ if padded, $0$ otherwise).
   - Sequence vector: $h_u = \sum_{k=1}^L \alpha_k E_{H, k}$.
3. **Genre Preference Projection:** Multi-hot genre affinity vector $g_u \in \mathbb{R}^{18}$ projected via dense layer $W_g \in \mathbb{R}^{18 \times D}$.
4. **Fusion MLP:**
   $$z_u = \text{LayerNorm}\left( \text{ReLU}\left( W_2 \cdot \text{ReLU}\left( W_1 [e_u \,\|\, h_u \,\|\, g_u] + b_1 \right) + b_2 \right) \right)$$
   $$\hat{u} = \frac{z_u}{\|z_u\|_2}$$

#### Item Tower $g_{\phi}(i, g_i, \text{stats}_i)$:
1. **Item Identity Embedding:** $e_i \in \mathbb{R}^D$.
2. **Item Genre Multi-Hot Projection:** $g_i \in \mathbb{R}^{18}$ projected to $D$ dimensions.
3. **Continuous Metadata Projection:** Release year, Bayesian rating, and log-activity count $[y_i, r_{\text{Bayes}, i}, \log(1+N_i)] \in \mathbb{R}^3$ projected to $D$ dimensions.
4. **Fusion MLP:**
   $$z_i = \text{LayerNorm}\left( \text{ReLU}\left( V_2 \cdot \text{ReLU}\left( V_1 [e_i \,\|\, \tilde{g}_i \,\|\, \tilde{s}_i] + c_1 \right) + c_2 \right) \right)$$
   $$\hat{i} = \frac{z_i}{\|z_i\|_2}$$

### 3.2 In-Batch Negative Contrastive Loss
Because real implicit feedback catalogs contain only observed interactions and no definitive negative impression logs, CinePulse treats other batch items as implicit negatives.
For batch size $B$, given positive pairs $(u_b, i_b)_{b=1}^B$:

$$\mathcal{L} = -\frac{1}{B} \sum_{b=1}^B \log \frac{\exp(\hat{u}_b \cdot \hat{i}_b / \tau)}{\sum_{j=1}^B \exp(\hat{u}_b \cdot \hat{i}_j / \tau)}$$

Where temperature $\tau = 0.05$. This forces positive items to have maximal cosine similarity while repelling unobserved items in the batch.

### 3.3 Exact Inner Product Indexing (FAISS)
Once trained, the item tower computes offline vectors for all 3,416 catalog movies:
$$\mathbf{V}_{\text{catalog}} = [\hat{i}_1, \hat{i}_2, \dots, \hat{i}_M] \in \mathbb{R}^{M \times D}$$

**Exact vs Approximate Nearest Neighbors (ANN):**
- For MovieLens 1M (3,416 catalog items), CinePulse utilizes `faiss.IndexFlatIP` (Exact Inner Product Search). Because the catalog fits in memory, brute-force cosine distance executes in **<3 ms** with 100% theoretical recall (zero quantization or pruning approximation loss).
- For web-scale industrial catalogs (e.g., MovieLens 32M+ or YouTube scale with 10M+ videos), CinePulse supports drop-in migration to `faiss.IndexIVFFlat` (Inverted File with coarse Voronoi quantizers) or `faiss.IndexHNSWFlat` (Hierarchical Navigable Small World graphs) for logarithmic approximate candidate retrieval.

---

## 4. Stage C: Personalized Ranking (LightGBM LambdaRank)

While Two-Tower models excel at filtering millions of items to candidates via inner products, their bilinear form prevents cross-feature interactions (e.g., does this user prefer *older* movies specifically when they are *Sci-Fi*?).

### 4.1 The 17-Dimensional Feature Engineering Matrix
For each user-candidate pair $(u, i)$, the ranker extracts 17 verified features:
1. `candidate_rank`: 0-indexed position within the multi-source candidate union.
2. `two_tower_score`: Dot-product similarity $\langle e_i, u \rangle$ from the Two-Tower neural model.
3. `cf_score`: Cosine similarity score from the Item-CF baseline.
4. `genre_overlap`: Multi-hot dot product between user history genre profile and movie genres.
5. `genre_cosine`: Cosine similarity between user genre profile and movie genres.
6. `log_rating_count`: Log-transformed interaction frequency $\log(1 + N_i)$.
7. `rating_mean`: Raw arithmetic average community rating.
8. `rating_std`: Rating standard deviation across community interactions.
9. `bayesian_rating`: Bayesian smoothed rating prior.
10. `release_year`: Year of film release.
11. `year_diff`: Absolute difference between film release year and user's historical mean year.
12. `user_activity_count`: Number of past user ratings in training history.
13. `is_source_two_tower`: Indicator whether item was retrieved by Two-Tower ANN search.
14. `is_source_item_cf`: Indicator whether item was retrieved by Item-CF.
15. `is_source_popularity`: Indicator whether item was retrieved by Popularity.
16. `cf_x_bayesian`: Interaction term between CF similarity and centered Bayesian rating.
17. `tt_x_genre`: Interaction term between Two-Tower score and genre cosine.

### 4.2 LambdaRank Objective & Empirical Impact
The LightGBM model is trained with the `lambdarank` objective with query groups structured per user session.
The LightGBM ranker boosts **NDCG@10 from 0.0346 to 0.0641 (+85.3%)** and **Recall@20 from 0.1060 to 0.1880 (+77.4%)** over standalone ANN retrieval.

---

## 5. Stage D: Diversity & Novelty Re-Ranking

A classic ranking model will recommend 10 superhero sequels if the user watched an action movie. This creates catastrophic user fatigue.

### 5.1 Maximal Marginal Relevance (MMR)
Let $R$ be the candidate set from Stage C, and $S$ be the selected recommendation list. At each step, the next item $i^*$ is selected greedily:

$$i^* = \arg\max_{i \in R \setminus S} \left[ \lambda \cdot \text{Score}(i) - (1 - \lambda) \cdot \max_{j \in S} \text{Sim}(i, j) + \beta \cdot \text{Novelty}(i) \right]$$

Where:
- $\text{Score}(i) \in (0, 1)$: Standardized sigmoid-normalized ranker score $\sigma\left(\frac{s_i - \mu_s}{\max(\sigma_s, 0.25)}\right)$ (replacing legacy min-max scaling to prevent artificial top-item score saturation).
- $\text{Sim}(i, j)$: Cosine similarity between multi-hot genre vectors of item $i$ and item $j$.
- $\lambda \in [0, 1]$: Diversity tradeoff slider (configurable dynamically by the user).
  - $\lambda = 1.0$: Pure relevance (highest ranking accuracy, lowest diversity).
  - $\lambda = 0.85$: Production default sweet spot ($\beta=0.03$).
  - $\lambda = 0.65$: High-discovery exploratory mode ($\beta=0.15$).
- $\text{Novelty}(i) = -\log_2 \left( \frac{N_i + 1}{\sum N} \right)$: Self-information / serendipity boost in bits.
- $\beta = 0.03$: Tuned weight assigned to long-tail discovery.

### 5.2 Empirical Diversity Impact
- **Without MMR ($\lambda = 1.0, \beta=0.0$):** Intra-list diversity is **0.5965**, with pure ranker relevance.
- **With Balanced MMR ($\lambda = 0.85, \beta = 0.03$):** Intra-list diversity rises to **0.6708 (+12.5% gain)**, with catalog coverage reaching **71.37%** (+70.5% wider than Item-CF's 41.86%) and NDCG@10 preserved at **0.0601**.

---

## 6. Stage E: Real-Time Session Personalization

### 6.1 The Cold-Start & Fast Adaptation Problem
Training deep neural networks is computationally expensive and cannot run synchronously on HTTP POST requests. However, users expect instant feedback:
- If a user dislikes a horror movie, the feed must **immediately** stop recommending horror movies.
- If a user saves an indie drama, related dramas should surface on the next refresh.

### 6.2 The Hybrid Session Vector Formulation
CinePulse implements an asynchronous session state engine:
1. **Suppression Set:** Any movie with `feedback_type in {'dislike', 'dismiss'}` is immediately added to the user's volatile session black-list $\mathcal{B}_u$. All future retrieval passes mask $\mathcal{B}_u$ in FAISS queries.
2. **Session Preference Shift:**
   Let $v_u^{(0)}$ be the user's base offline representation.
   When the user interacts with movie $m_k$ with embedding $e_{m_k}$ and valence $w_k$ (+1.0 for like, +0.7 for save, -1.0 for dislike):
   $$v_u^{(t)} = (1 - \gamma) v_u^{(t-1)} + \gamma \sum_k w_k e_{m_k}$$
   where $\gamma = 0.25$ is the session decay factor.
3. **Execution Latency:**
   Session updates occur in **<0.5 ms** in-memory. The subsequent recommendation request uses $v_u^{(t)}$ for FAISS candidate retrieval, producing real-time personalization with zero GPU gradient overhead.

---

## 7. Serving Infrastructure & Observability

- **FastAPI Asynchronous Gateway:** Native coroutines for non-blocking I/O, SQLite persistence via `aiosqlite` and SQLAlchemy 2.0.
- **Latency Budget:**
  - Candidate Retrieval (ANN + ItemCF + Pop): ~2.96 ms
  - Feature Extraction & LightGBM Scoring: ~12.27 ms
  - MMR Re-Ranking (20 items): ~0.70 ms
  - Database Session & Audit Persistence: ~1.50 ms
  - **Total P95 Latency: 20.88 ms (<25 ms SLA)**
- **Observability Endpoints:**
  - `/health`: Hardware inspection (CUDA device, VRAM allocation, model checkpoints).
  - `/api/metrics`: Live empirical evaluation matrix comparing all system stages.
  - `/api/users/{user_id}/insights`: Real-time breakdown of user genre affinities and recommendation source attributions.
