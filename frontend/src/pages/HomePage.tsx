import { useState } from "react";
import type { RecommendationFeed } from "../types";
import { isStaticHost } from "../api/client";
import { HeroBanner } from "../components/HeroBanner";
import { MovieRail } from "../components/MovieRail";
import { EmptyState, Skeletons } from "../components/Primitives";
export function HomePage({
  feed,
  loading,
  onRefresh,
  onOpenDetail,
  activeProfileName,
}: {
  feed: RecommendationFeed | null;
  loading: boolean;
  onRefresh: (diversity?: number, novelty?: number) => void;
  onOpenDetail: (id: number) => void;
  activeProfileName: string;
}) {
  const [diversity, setDiversity] = useState(0.85);
  const [novelty, setNovelty] = useState(0.03);
  const hero =
    feed?.hero_featured || Object.values(feed?.rails || {}).flat()[0];
  const heroExplanation = Object.values(feed?.rails || {})
    .flat()
    .find((movie) => movie.movie_idx === hero?.movie_idx)?.reason;
  return (
    <>
      {hero && (
        <HeroBanner
          movie={{ ...hero, reason: heroExplanation || hero.reason }}
          onOpenDetail={onOpenDetail}
        />
      )}
      <div className="content-width home-rails">
        <div className="collection-intro">
          <div>
            <span className="eyebrow">A LITTLE MORE YOU</span>
            <h2>Find your next favorite.</h2>
          </div>
          <p>
            Selected for {activeProfileName.split(" ")[0]}.<br /> Stay curious.
            There’s a whole world of cinema.
          </p>
        </div>
        {loading && !feed ? (
          <Skeletons />
        ) : feed ? (
          Object.entries(feed.rails).map(([title, movies], i) => (
            <MovieRail
              key={title}
              title={
                title === "Top Picks For You" ? "Recommended for you" : title
              }
              subtitle={
                i === 0 ? "Good stories, chosen around your taste." : undefined
              }
              movies={movies}
              onOpenDetail={onOpenDetail}
            />
          ))
        ) : (
          <EmptyState title="Your discoveries couldn’t load.">
            <button className="button secondary" onClick={() => onRefresh()}>
              Try again
            </button>
          </EmptyState>
        )}
        {!isStaticHost && (
          <details className="technical">
            <summary>Fine-tune live recommendations</summary>
            <label>
              Diversity balance{" "}
              <input
                aria-label="Diversity balance"
                type="range"
                min="0"
                max="1"
                step=".05"
                value={diversity}
                onChange={(e) => setDiversity(+e.target.value)}
              />
            </label>
            <label>
              Novelty{" "}
              <input
                aria-label="Novelty"
                type="range"
                min="0"
                max=".3"
                step=".01"
                value={novelty}
                onChange={(e) => setNovelty(+e.target.value)}
              />
            </label>
            <button
              className="button secondary"
              disabled={loading}
              onClick={() => onRefresh(diversity, novelty)}
            >
              Update recommendations
            </button>
          </details>
        )}
      </div>
    </>
  );
}
