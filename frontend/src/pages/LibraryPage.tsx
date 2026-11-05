import { useEffect, useState } from "react";
import { Bookmark } from "lucide-react";
import type { RecommendationItem, UserProfile } from "../types";
import { loadMovies } from "../api/client";
import { useMovies } from "../state";
import { MovieCard } from "../components/MovieCard";
import { EmptyState, Skeletons } from "../components/Primitives";
export function LibraryPage({
  currentUser,
  onOpenDetail,
  onExplore,
}: {
  currentUser: UserProfile | null;
  onOpenDetail: (id: number) => void;
  onExplore: () => void;
}) {
  const { saved } = useMovies();
  const [movies, setMovies] = useState<RecommendationItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  useEffect(() => {
    let valid = true;
    loadMovies()
      .then((m) => {
        if (valid) setMovies(m);
      })
      .catch(() => {
        if (valid) setError(true);
      })
      .finally(() => {
        if (valid) setLoading(false);
      });
    return () => {
      valid = false;
    };
  }, []);
  const items = saved.flatMap((id) => movies.filter((m) => m.movie_idx === id));
  return (
    <div className="content-width page">
      <div className="page-heading">
        <span className="eyebrow">FOR ANOTHER EVENING</span>
        <h1>Your list. Your kind of cinema.</h1>
        <p>
          A collection of possibilities, saved for{" "}
          {currentUser?.display_name.split(" ")[0] || "you"}.
        </p>
      </div>
      <div className="catalog-toolbar">
        <span className="inline-label">
          <Bookmark size={18} /> Watchlist{" "}
          <span className="muted">/ {items.length} films</span>
        </span>
        <span className="muted">Saved in this browser</span>
      </div>
      {loading ? (
        <Skeletons />
      ) : error ? (
        <EmptyState title="Your collection could not load.">
          Please reload to try again.
        </EmptyState>
      ) : items.length ? (
        <div className="movie-grid">
          {items.map((m) => (
            <MovieCard
              key={m.movie_idx}
              movie={m}
              onOpenDetail={onOpenDetail}
            />
          ))}
        </div>
      ) : (
        <EmptyState title="Make room for great films.">
          Tap the + on any film to save it for later.
          <br />
          <button className="button primary" onClick={onExplore}>
            Explore the collection
          </button>
        </EmptyState>
      )}
    </div>
  );
}
