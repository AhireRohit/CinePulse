import benchmark from "../data/benchmark.json";
import { Pipeline } from "../components/Primitives";
const rows = benchmark.models;
const first = rows[0].metrics;
const second = rows[1].metrics;
const experiments = [
  [
    "Collaborative filtering",
    "Tested neighborhood size, shrinkage, and weighting to improve co-rating signals.",
  ],
  [
    "Candidate retrieval",
    "Expanded the neural candidate pool to give the ranker more possibilities.",
  ],
  [
    "Ranking features",
    "Audited and expanded the ranker from 17 to 24 causal features.",
  ],
  [
    "Two-Tower variants",
    "Compared neural retrieval configurations and training choices.",
  ],
  [
    "Score fusion",
    "Compared ranker and retrieval blends; the promoted configuration omits fusion.",
  ],
  [
    "Diversity reranking",
    "Selected a relevance-first balance with MMR λ = 0.95 and novelty β = 0.01.",
  ],
];
export function EvaluationPage() {
  return (
    <div className="content-width page benchmark-page">
      <div className="page-heading">
        <span className="eyebrow">THE CINEPULSE ENGINEERING STORY</span>
        <h1>
          Better recommendations.
          <br />
          <span className="muted">Measured, not assumed.</span>
        </h1>
        <p>From a working baseline to a more relevant discovery engine.</p>
      </div>
      <div className="editorial-note">
        <span className="status-dot" />
        <p>
          Offline test results · 3,974 held-out users · MovieLens 1M · No live
          browser benchmark
        </p>
      </div>
      <section className="benchmark-lead">
        <div>
          <span className="eyebrow">01 / A MEASURABLE STEP FORWARD</span>
          <h2>
            Put better films
            <br />
            closer to the top.
          </h2>
          <p>
            NDCG rewards placing a relevant film higher in the list. V2 improves
            from <strong>0.0560</strong> to <strong>0.0778</strong> on the same
            test cohort.
          </p>
          <span className="big-lift">
            +38.9%<small>NDCG@10 relative improvement</small>
          </span>
        </div>
        <div
          className="comparison-chart"
          role="img"
          aria-label="NDCG at 10: V1 0.0560, V2 0.0778"
        >
          <div className="chart-top">
            <span>RANKING QUALITY</span>
            <span>NDCG@10 ↑</span>
          </div>
          <div className="comparison-row">
            <span>V1 / Baseline</span>
            <div>
              <i style={{ width: "64%" }} />
              <strong>0.0560</strong>
            </div>
          </div>
          <div className="comparison-row winner">
            <span>V2 / Production</span>
            <div>
              <i style={{ width: "89%" }} />
              <strong>0.0778</strong>
            </div>
          </div>
          <p>
            Paired bootstrap 95% CI for the improvement:
            <br />
            <strong>+0.0165 to +0.0277</strong> · p = 0.001
          </p>
        </div>
      </section>
      <div className="benchmark-stats">
        <div>
          <span>Recall@20</span>
          <strong>
            22.19<small>%</small>
          </strong>
          <p>Up from 18.09% in V1</p>
        </div>
        <div>
          <span>Coverage@100</span>
          <strong>
            95.08<small>%</small>
          </strong>
          <p>Catalog reached across test users</p>
        </div>
        <div>
          <span>V2 P95 latency</span>
          <strong>
            27.92<small>ms</small>
          </strong>
          <p>Recorded offline pipeline latency</p>
        </div>
      </div>
      <section className="story-section">
        <span className="eyebrow">02 / THE ARCHITECTURE</span>
        <h2>Three stages. One great next watch.</h2>
        <Pipeline />
      </section>
      <section className="story-section">
        <span className="eyebrow">03 / SIX LINES OF INQUIRY</span>
        <h2>Progress through experimentation.</h2>
        <p>
          Six experiment families informed selection on the validation cohort.
          The headline results above come exclusively from the separate
          3,974-user test cohort.
        </p>
        <div className="experiment-grid">
          {experiments.map(([title, text], i) => (
            <article key={title}>
              <span className="step-number">0{i + 1}</span>
              <h3>{title}</h3>
              <p>{text}</p>
            </article>
          ))}
        </div>
        <p className="source-note">
          Validation involved 51 configurations; 800 validation users were used
          for early stopping. Validation selection bias is documented in the
          experiment report.
        </p>
      </section>
      <section className="story-section tradeoff">
        <div>
          <span className="eyebrow">04 / THE HONEST TRADEOFF</span>
          <h2>
            More relevance.
            <br />
            Less genre variety.
          </h2>
          <p>
            V2 prioritizes personal relevance. Intra-list genre diversity fell
            from <strong>0.6933</strong> to <strong>0.5165</strong> (25.5%
            lower), while catalog coverage expanded from 72.57% to 95.08%. A
            wider catalog does not mean more variety within each person’s list.
          </p>
        </div>
        <div className="tradeoff-bars">
          {[
            ["Ranking quality", first.ndcg_at_10, second.ndcg_at_10, 0.1],
            [
              "Genre diversity",
              first.intra_list_diversity_at_20,
              second.intra_list_diversity_at_20,
              1,
            ],
            [
              "Catalog coverage",
              first.catalog_coverage_at_100,
              second.catalog_coverage_at_100,
              1,
            ],
          ].map(([label, a, b, max]) => (
            <div key={String(label)}>
              <strong>{label}</strong>
              <span>
                <i style={{ width: `${(Number(a) / Number(max)) * 100}%` }} />
                V1
              </span>
              <span className="v2">
                <i style={{ width: `${(Number(b) / Number(max)) * 100}%` }} />
                V2
              </span>
            </div>
          ))}
        </div>
      </section>
      <section className="story-section">
        <span className="eyebrow">05 / THE FULL PICTURE</span>
        <h2>Every number, in context.</h2>
        <p>
          All rows below use the same 3,974-user sequestered test cohort.
          Missing fields are never filled with estimates.
        </p>
        <details className="technical" open>
          <summary>Full benchmark table · held-out test cohort</summary>
          <div
            className="table-scroll"
            tabIndex={0}
            role="region"
            aria-label="Scrollable full benchmark table"
          >
            <table>
              <thead>
                <tr>
                  <th>Configuration</th>
                  {[
                    "NDCG@10",
                    "Recall@20",
                    "Recall@100",
                    "MRR@10",
                    "Coverage@100",
                    "Diversity",
                    "P95 latency",
                  ].map((s) => (
                    <th key={s}>{s}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => (
                  <tr key={r.name}>
                    <th>{r.name}</th>
                    <td>{r.metrics.ndcg_at_10.toFixed(4)}</td>
                    <td>{(r.metrics.recall_at_20 * 100).toFixed(2)}%</td>
                    <td>{(r.metrics.recall_at_100 * 100).toFixed(2)}%</td>
                    <td>{r.metrics.mrr_at_10.toFixed(4)}</td>
                    <td>
                      {(r.metrics.catalog_coverage_at_100 * 100).toFixed(2)}%
                    </td>
                    <td>{r.metrics.intra_list_diversity_at_20.toFixed(4)}</td>
                    <td>{r.metrics.p95_latency_ms.toFixed(2)} ms</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </details>
        <details className="technical">
          <summary>Methodology & source artifacts</summary>
          <p>
            Temporal leave-last-two-out splitting. Test results were confirmed
            after model selection. Bootstrap resolution is approximately 0.001
            with 1,000 resamples. The V2 versus pure-ranker NDCG confidence
            interval includes zero; no significance claim is made for that
            comparison.
          </p>
          <p>
            Source: {benchmark.source}. Experiment narrative:
            artifacts/experiments/relevance_v2/EXPERIMENT_REPORT.md. Values are
            exported unchanged from the original confirmation artifact.
          </p>
        </details>
      </section>
    </div>
  );
}
