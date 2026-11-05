import { artwork, displayTitle } from "../moviePresentation";
import { ArrowUpRight } from "lucide-react";
import type { RecommendationItem } from "../types";
import { Poster, Rating, SaveButton } from "./Primitives";
export function HeroBanner({
  movie,
  onOpenDetail,
}: {
  movie: RecommendationItem;
  onOpenDetail: (id: number) => void;
}) {
  const backdrop = artwork(movie);
  return (
    <section className={`hero ${backdrop ? "" : "hero-poster"}`}>
      <div className="hero-art" aria-hidden="true">
        {backdrop ? (
          <img src={backdrop} alt="" />
        ) : (
          <Poster url={movie.poster_url} title={movie.title} eager />
        )}
      </div>
      <div className="hero-shade" />
      <div className="hero-content">
        <span className="eyebrow">
          <span className="accent-line" /> YOUR NEXT GREAT WATCH
        </span>
        <h1>{displayTitle(movie.title)}</h1>
        <div className="hero-meta">
          <span>{movie.release_year}</span>
          <span className="meta-dot" />
          {movie.genres.slice(0, 3).join(" · ")}
          <Rating value={movie.rating_mean} />
        </div>
        <p className="hero-reason">
          {movie.reason.replace(/^Top Spotlight:\s*/, "")}
        </p>
        <div className="hero-actions">
          <button
            className="button primary"
            onClick={() => onOpenDetail(movie.movie_idx)}
          >
            View Details <ArrowUpRight size={18} />
          </button>
          <SaveButton id={movie.movie_idx} />
        </div>
      </div>
      <div className="hero-caption">
        <span>THE SPOTLIGHT</span>
        <span>A film worth your evening.</span>
      </div>
    </section>
  );
}
