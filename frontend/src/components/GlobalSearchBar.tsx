import { displayTitle } from "../moviePresentation";
import { useEffect, useState } from "react";
import { Search, ArrowUpRight } from "lucide-react";
import { searchMovies } from "../api/client";
import type { RecommendationItem } from "../types";
import { Dialog, Poster, Rating } from "./Primitives";
export function GlobalSearchBar({
  onClose,
  onOpenDetail,
}: {
  onClose: () => void;
  onOpenDetail: (id: number) => void;
}) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<RecommendationItem[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(false);
  useEffect(() => {
    let valid = true;
    if (!query.trim()) {
      setResults([]);
      setBusy(false);
      return;
    }
    setBusy(true);
    const timer = setTimeout(() => {
      searchMovies(query)
        .then((r) => {
          if (valid) {
            setResults(r);
            setError(false);
          }
        })
        .catch(() => {
          if (valid) setError(true);
        })
        .finally(() => {
          if (valid) setBusy(false);
        });
    }, 200);
    return () => {
      valid = false;
      clearTimeout(timer);
    };
  }, [query]);
  return (
    <Dialog title="Search films" onClose={onClose} className="search-dialog">
      <div className="dialog-padding">
        <span className="eyebrow">FIND YOUR NEXT STORY</span>
        <h2>Search the collection.</h2>
        <label className="search-field">
          <Search size={20} />
          <input
            autoFocus
            placeholder="Search by title, genre, or year"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <kbd>ESC</kbd>
        </label>
        <div className="search-results" aria-live="polite">
          {busy ? (
            <p className="muted">Searching films…</p>
          ) : error ? (
            <p role="alert">Search is unavailable. Try again.</p>
          ) : !query ? (
            <p className="muted">Try “Matrix”, “Film-Noir”, or “1995”.</p>
          ) : !results.length ? (
            <p className="muted">No films found. Try a different title.</p>
          ) : (
            results.map((m) => (
              <button
                className="search-result"
                key={m.movie_idx}
                onClick={() => {
                  onClose();
                  onOpenDetail(m.movie_idx);
                }}
              >
                <Poster url={m.poster_url} title={m.title} />
                <span>
                  <strong>{displayTitle(m.title)}</strong>
                  <small>
                    {m.release_year} · {m.genres.join(" / ")}
                  </small>
                </span>
                <Rating value={m.rating_mean} />
                <ArrowUpRight size={18} />
              </button>
            ))
          )}
        </div>
      </div>
    </Dialog>
  );
}
