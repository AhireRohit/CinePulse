import { useEffect, useState } from "react";
import type { UserProfile, UserInsights } from "../types";
import { getUserInsights, isStaticHost } from "../api/client";
import { EmptyState, Pipeline } from "../components/Primitives";
const sourceNames: Record<string, string> = {
  two_tower: "Neural discovery",
  item_cf: "Similar audience tastes",
  popularity: "Community favorites",
};
export function InsightsPage({
  currentUser,
}: {
  currentUser: UserProfile | null;
}) {
  const [data, setData] = useState<UserInsights | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    let valid = true;
    setData(null);
    if (currentUser)
      getUserInsights(currentUser.user_idx)
        .then((d) => {
          if (valid) {
            setData(d);
            setError(false);
          }
        })
        .catch(() => {
          if (valid) setError(true);
        });
    return () => {
      valid = false;
    };
  }, [currentUser]);
  const custom =
    !!currentUser && ![12, 45, 88, 120].includes(currentUser.user_idx);
  const sources = Object.entries(data?.retrieval_sources_breakdown || {});
  const total = sources.reduce((a, [, v]) => a + v, 0);
  let offset = 0;
  const stops = sources
    .map(([, v], i) => {
      const start = offset;
      offset += (v / Math.max(total, 1)) * 100;
      return `var(--chart-${i % 3}) ${start}% ${offset}%`;
    })
    .join(",");
  return (
    <div className="content-width page">
      <div className="page-heading">
        <span className="eyebrow">BEHIND YOUR DISCOVERIES</span>
        <h1>Good taste meets good science.</h1>
        <p>
          A closer look at how CinePulse finds films for{" "}
          {currentUser?.display_name.split(" ")[0] || "you"}.
        </p>
      </div>
      <div className="editorial-note">
        <span className="status-dot" />
        <p>
          {isStaticHost
            ? custom
              ? "Your profile uses local curation: chosen genres and community ratings. No Two-Tower inference is performed for this profile."
              : "A snapshot of the exported V2 recommendation pipeline. These are recorded results, not live browser inference."
            : "Insights from the connected recommendation engine."}
        </p>
      </div>
      {error ? (
        <EmptyState title="Insights couldn’t load.">
          Please reload to try again.
        </EmptyState>
      ) : !data ? (
        <div
          className="skeleton chart-skeleton"
          aria-label="Loading insights"
        />
      ) : (
        <>
          <div className="insight-grid">
            <section className="chart-panel">
              <span className="eyebrow">01 / YOUR STARTING POINT</span>
              <h2>
                {isStaticHost
                  ? "The genres you gravitate to"
                  : "Your genre affinity"}
              </h2>
              <p>
                {isStaticHost
                  ? "Declared profile preferences. Equal bars indicate selected genres, not learned affinity."
                  : "Genre weights from your recommendation profile."}
              </p>
              <div className="genre-bars">
                {data.top_affinity_genres.map(({ genre, affinity }) => (
                  <div key={genre}>
                    <div>
                      <span>{genre}</span>
                      <span className="muted">
                        {isStaticHost
                          ? "Selected"
                          : `${(affinity * 100).toFixed(0)}%`}
                      </span>
                    </div>
                    <div className="bar-track">
                      <span
                        style={{
                          width: `${isStaticHost ? 100 : affinity * 100}%`,
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </section>
            <section className="chart-panel">
              <span className="eyebrow">02 / A WIDER POINT OF VIEW</span>
              <h2>Where discoveries begin</h2>
              <p>
                {custom && isStaticHost
                  ? "Films are filtered by your genres, then ordered by community ratings."
                  : "Three complementary ways to build a pool of possibilities."}
              </p>
              {total > 0 ? (
                <div className="source-chart">
                  <div
                    className="donut"
                    role="img"
                    aria-label={sources
                      .map(([k, v]) => `${sourceNames[k] || k}: ${v}`)
                      .join(", ")}
                    style={{ background: `conic-gradient(${stops})` }}
                  >
                    <div>
                      <strong>{total}</strong>
                      <span>candidates</span>
                    </div>
                  </div>
                  <div className="source-legend">
                    {sources.map(([k, v], i) => (
                      <div key={k}>
                        <i style={{ background: `var(--chart-${i % 3})` }} />
                        <span>{sourceNames[k] || k}</span>
                        <strong>{v}</strong>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="local-curation">
                  <strong>Genre → Community rating → Your discoveries</strong>
                  <p>Your saved and hidden films stay local to this profile.</p>
                </div>
              )}
            </section>
          </div>
          <section className="story-section">
            <span className="eyebrow">THE BIG PICTURE</span>
            <h2>A thoughtful path to your next film.</h2>
            <p>
              {custom && isStaticHost
                ? "The deployed backend uses the pipeline below. Your local profile uses genre-based curation."
                : "The backend brings together discovery, personal relevance, and a little surprise."}
            </p>
            <Pipeline />
          </section>
          <details className="technical">
            <summary>Explore the technical details</summary>
            <p>
              Two-Tower embeddings retrieve candidates through FAISS. Item-based
              collaborative filtering adds co-rating signals. LightGBM
              LambdaRank scores the combined pool, followed by Maximal Marginal
              Relevance reranking.
            </p>
            <p>
              Recorded training interactions: {data.total_interactions}. Local
              likes: {data.active_likes_count}. Local saves:{" "}
              {data.active_saves_count}.
            </p>
            {Object.keys(data.timing_breakdown_ms).length > 0 && (
              <>
                <h3>
                  {isStaticHost ? "Recorded export timings" : "Backend timings"}
                </h3>
                <p>
                  Export runs can include cold-start overhead; use Benchmarks
                  for the documented offline P95 measurement.
                </p>
                {Object.entries(data.timing_breakdown_ms).map(([k, v]) => (
                  <p key={k}>
                    {k.replaceAll("_", " ")}: {v.toFixed(2)} ms
                  </p>
                ))}
              </>
            )}
          </details>
        </>
      )}
    </div>
  );
}
