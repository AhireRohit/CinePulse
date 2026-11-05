import { displayTitle } from "../moviePresentation";
import { useEffect, useMemo, useState } from "react";
import { Search, LayoutGrid, List, ArrowUpRight } from "lucide-react";
import { loadMovies, searchMovies, isStaticHost } from "../api/client";
import type { RecommendationItem } from "../types";
import { MovieCard } from "../components/MovieCard";
import { EmptyState, Poster, Rating, SaveButton, Skeletons } from "../components/Primitives";
export function ExplorePage({
  onOpenDetail,
}: {
  onOpenDetail: (id: number) => void;
}) {
  const [catalog, setCatalog] = useState<RecommendationItem[]>([]);
  const [query, setQuery] = useState("");
  const [genre, setGenre] = useState("");
  const [era, setEra] = useState("");
  const [sort, setSort] = useState("popular");
  const [list, setList] = useState(false);
  const [limit, setLimit] = useState(36);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);
  useEffect(() => {
    let valid = true;
    setLoading(true);
    const timer = setTimeout(() => {
      (isStaticHost || !query.trim() ? loadMovies() : searchMovies(query))
        .then((data) => {
          if (valid) {
            setCatalog(data);
            setError(false);
          }
        })
        .catch(() => {
          if (valid) setError(true);
        })
        .finally(() => {
          if (valid) setLoading(false);
        });
    }, 180);
    return () => {
      valid = false;
      clearTimeout(timer);
    };
  }, [query]);
  useEffect(() => {
    setLimit(36);
  }, [query, genre, era, sort]);
  const genres = [
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
    "Documentary",
  ];
  const results = useMemo(() => {
    const q = query.toLowerCase().trim();
    return catalog
      .filter(
        (m) =>
          (!q ||
            [
              m.title,
              m.full_title,
              String(m.release_year),
              ...(m.genres || []),
            ].some((s) => s?.toLowerCase().includes(q))) &&
          (!genre || m.genres.includes(genre)) &&
          (!era ||
            (era === "classic"
              ? m.release_year < 1970
              : m.release_year >= +era && m.release_year < +era + 10)),
      )
      .sort((a, b) =>
        sort === "rating"
          ? b.rating_mean - a.rating_mean
          : sort === "year"
            ? b.release_year - a.release_year
            : (b.rating_count || 0) - (a.rating_count || 0),
      );
  }, [catalog, query, genre, era, sort]);
  const reset = () => {
    setQuery("");
    setGenre("");
    setEra("");
  };
  return (
    <div className="content-width page">
      <div className="page-heading">
        <span className="eyebrow">FOLLOW YOUR CURIOSITY</span>
        <h1>A world of cinema.</h1>
        <p>
          From the familiar to the unexpected. Find a film that stays with you.
        </p>
      </div>
      <label className="search-field catalog-search">
        <Search size={21} />
        <input
          aria-label="Search catalog"
          placeholder="Search by title, genre, or year…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <span className="muted">THE COLLECTION</span>
      </label>
      <div className="genre-filters" aria-label="Genre filters">
        {["", ...genres].map((g) => (
          <button
            key={g}
            aria-pressed={genre === g}
            className={genre === g ? "selected" : ""}
            onClick={() => setGenre(g)}
          >
            {g || "All films"}
          </button>
        ))}
      </div>
      <div className="catalog-toolbar">
        <span aria-live="polite">
          <strong>{results.length.toLocaleString()}</strong> films
          {genre && ` in ${genre}`}
        </span>
        <div className="catalog-controls">
          <label>
            <span className="sr-only">Release era</span>
            <select
              aria-label="Release era"
              value={era}
              onChange={(e) => setEra(e.target.value)}
            >
              <option value="">All eras</option>
              <option value="classic">Before 1970</option>
              {["1970", "1980", "1990", "2000"].map((y) => (
                <option key={y} value={y}>
                  {y}s
                </option>
              ))}
            </select>
          </label>
          <select
            aria-label="Sort movies"
            value={sort}
            onChange={(e) => setSort(e.target.value)}
          >
            <option value="popular">Most popular</option>
            <option value="rating">Highest rated</option>
            <option value="year">Newest first</option>
          </select>
          <div className="view-toggle">
            <button
              className={!list ? "active" : ""}
              aria-label="Grid view"
              aria-pressed={!list}
              onClick={() => setList(false)}
            >
              <LayoutGrid size={18} />
            </button>
            <button
              className={list ? "active" : ""}
              aria-label="List view"
              aria-pressed={list}
              onClick={() => setList(true)}
            >
              <List size={18} />
            </button>
          </div>
        </div>
      </div>
      {loading ? (
        <Skeletons />
      ) : error ? (
        <EmptyState title="The collection couldn’t load.">
          Please reload the page to try again.
        </EmptyState>
      ) : !results.length ? (
        <EmptyState title="No films in this frame.">
          Try another search or{" "}
          <button className="text-button" onClick={reset}>
            clear your filters
          </button>
          .
        </EmptyState>
      ) : (
        <div className={list ? "catalog-list" : "movie-grid"}>
          {results.slice(0, limit).map((m) =>
            list ? (
              <article className="catalog-row" key={m.movie_idx}>
                <button
                  className="row-poster"
                  onClick={() => onOpenDetail(m.movie_idx)}
                  aria-label={`View ${m.title}`}
                >
                  <Poster url={m.poster_url} title={m.title} />
                </button>
                <button
                  className="row-title"
                  onClick={() => onOpenDetail(m.movie_idx)}
                >
                  <strong>{displayTitle(m.title)}</strong>
                  <small>
                    {m.release_year} · {m.genres.join(" / ")}
                  </small>
                </button>
                <Rating value={m.rating_mean} />
                <SaveButton id={m.movie_idx} compact />
                <button
                  className="icon-button"
                  aria-label={`View details for ${m.title}`}
                  onClick={() => onOpenDetail(m.movie_idx)}
                >
                  <ArrowUpRight size={18} />
                </button>
              </article>
            ) : (
              <MovieCard
                key={m.movie_idx}
                movie={m}
                onOpenDetail={onOpenDetail}
              />
            ),
          )}
        </div>
      )}
      {limit < results.length && !loading && (
        <div className="load-more">
          <button
            className="button secondary"
            onClick={() => setLimit((l) => l + 36)}
          >
            Discover more films
          </button>
          <p className="muted">
            Showing {Math.min(limit, results.length)} of{" "}
            {results.length.toLocaleString()}
          </p>
        </div>
      )}
    </div>
  );
}
