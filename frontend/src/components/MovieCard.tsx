import { displayTitle } from "../moviePresentation";
import type { RecommendationItem } from "../types";
import { Poster, Rating, SaveButton } from "./Primitives";
interface Props {
  movie: RecommendationItem;
  onOpenDetail: (id: number) => void;
  onFeedback?: (
    id: number,
    action: "like" | "dislike" | "save" | "dismiss",
  ) => void;
}
export function MovieCard({ movie, onOpenDetail }: Props) {
  return (
    <article className="movie-card" data-testid="movie-card">
      <div className="poster-wrap">
        <button
          className="poster-open"
          aria-label={`View details for ${displayTitle(movie.title)}`}
          onClick={() => onOpenDetail(movie.movie_idx)}
        >
          <Poster url={movie.poster_url} title={movie.title} />
          <span className="poster-details">View details ↗</span>
        </button>
        <SaveButton id={movie.movie_idx} compact />
      </div>
      <button
        className="movie-title"
        onClick={() => onOpenDetail(movie.movie_idx)}
      >
        {displayTitle(movie.title)}
      </button>
      <div className="movie-meta">
        <span>{movie.release_year}</span>
        <Rating value={movie.rating_mean} />
      </div>
    </article>
  );
}
