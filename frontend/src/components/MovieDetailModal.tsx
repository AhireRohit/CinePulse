import { artwork, displayTitle } from "../moviePresentation";
import { useEffect, useState } from "react";
import { ThumbsUp, ThumbsDown, Star } from "lucide-react";
import type { MovieDetail, RecommendationItem } from "../types";
import { getMovie, isStaticHost } from "../api/client";
import { useMovies } from "../state";
import { Dialog, Poster, Rating, SaveButton } from "./Primitives";
import { MovieRail } from "./MovieRail";
export function MovieDetailModal({
  movieId,
  onClose,
  recommendation,
  onOpenDetail,
}: {
  movieId: number | null;
  onClose: () => void;
  recommendation?: RecommendationItem;
  onOpenDetail: (id: number) => void;
}) {
  const [movie, setMovie] = useState<MovieDetail | null>(null);
  const [error, setError] = useState(false);
  const { feedback, liked, disliked, ratings } = useMovies();
  useEffect(() => {
    let valid = true;
    setMovie(null);
    setError(false);
    if (movieId !== null)
      getMovie(movieId)
        .then((m) => {
          if (valid) setMovie(m);
        })
        .catch(() => {
          if (valid) setError(true);
        });
    return () => {
      valid = false;
    };
  }, [movieId]);
  if (movieId === null) return null;
  return (
    <Dialog
      title={movie ? displayTitle(movie.title) : "Movie details"}
      onClose={onClose}
      className="movie-dialog"
    >
      {!movie ? (
        <div className="dialog-padding">
          <h2>{error ? "This film couldn’t load." : "Finding your film…"}</h2>
          {error && (
            <button className="button secondary" onClick={onClose}>
              Back to discovering
            </button>
          )}
        </div>
      ) : (
        <>
          <div className="detail-backdrop">
            {artwork(movie) ? (
              <img src={artwork(movie)} alt="" />
            ) : (
              <div className="detail-poster-art">
                <Poster url={movie.poster_url} title={movie.title} />
              </div>
            )}
            <div />
          </div>
          <div className="detail-body">
            <div className="detail-poster">
              <Poster url={movie.poster_url} title={movie.title} eager />
            </div>
            <div className="detail-copy">
              <span className="eyebrow">THE CINEPULSE COLLECTION</span>
              <h2>{displayTitle(movie.title)}</h2>
              <div className="detail-meta">
                {movie.release_year}
                <span>·</span>
                {movie.genres.join(" / ")}
              </div>
              <div className="detail-rating">
                <Rating value={movie.rating_mean} />
                <span className="muted">
                  {movie.rating_count.toLocaleString()} community ratings
                </span>
              </div>
              {movie.overview && <p className="synopsis">{movie.overview}</p>}
              <div className="detail-actions">
                <SaveButton id={movie.movie_idx} />
                <button
                  className="button secondary"
                  aria-pressed={liked.includes(movie.movie_idx)}
                  onClick={() => void feedback(movie.movie_idx, "like")}
                >
                  <ThumbsUp size={17} />
                  {liked.includes(movie.movie_idx) ? "Liked" : "Like"}
                </button>
                <button
                  className="icon-button"
                  aria-label="Dislike film"
                  aria-pressed={disliked.includes(movie.movie_idx)}
                  onClick={() => void feedback(movie.movie_idx, "dislike")}
                >
                  <ThumbsDown size={18} />
                </button>
              </div>
              <div className="personal-rating">
                <span>Your rating</span>
                <div role="group" aria-label="Your rating">
                  {[1, 2, 3, 4, 5].map((n) => (
                    <button
                      aria-label={`Rate ${n} stars`}
                      aria-pressed={ratings[movie.movie_idx] === n}
                      key={n}
                      onClick={() => void feedback(movie.movie_idx, "rate", n)}
                    >
                      <Star
                        size={22}
                        fill={
                          (ratings[movie.movie_idx] || 0) >= n
                            ? "currentColor"
                            : "none"
                        }
                      />
                    </button>
                  ))}
                </div>
              </div>
              <details className="technical">
                <summary>Why this movie?</summary>
                <p>
                  {recommendation?.reason ||
                    "Opened from the film collection. No personalized recommendation explanation is available for this selection."}
                </p>
                {recommendation && (
                  <p>
                    Candidate source:{" "}
                    {recommendation.retrieval_source.replaceAll("_", " ")}.
                  </p>
                )}
                <p>
                  {isStaticHost
                    ? "Demo profile rankings come from exported V2 results. New profiles use local genre curation. Feedback is stored in this browser."
                    : "Recommendations are served by the connected recommendation backend."}
                </p>
              </details>
            </div>
          </div>
          {movie.similar_movies?.length > 0 && (
            <div className="detail-similar">
              <MovieRail
                title={isStaticHost ? "More in these genres" : "Similar films"}
                movies={movie.similar_movies}
                onOpenDetail={onOpenDetail}
              />
            </div>
          )}
        </>
      )}
    </Dialog>
  );
}
