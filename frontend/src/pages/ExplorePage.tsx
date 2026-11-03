import React, { useState, useEffect } from 'react';
import { Search, Compass } from 'lucide-react';
import type { RecommendationItem } from '../types';
import { searchMovies } from '../api/client';
import { MovieCard } from '../components/MovieCard';

interface ExplorePageProps {
  onOpenDetail: (movieId: number) => void;
  onFeedback: (movieId: number, action: 'like' | 'dislike' | 'save' | 'dismiss') => void;
}

const GENRES = [
  'All', 'Sci-Fi', 'Crime', 'Drama', 'Film-Noir', 'Mystery', 'Animation',
  'Action', 'Thriller', 'Comedy', 'Adventure', 'Romance', 'Horror', 'War'
];

export const ExplorePage: React.FC<ExplorePageProps> = ({
  onOpenDetail,
  onFeedback,
}) => {
  const [query, setQuery] = useState('');
  const [selectedGenre, setSelectedGenre] = useState('All');
  const [results, setResults] = useState<RecommendationItem[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const term = query.trim() || (selectedGenre === 'All' ? 'star' : selectedGenre);
    setLoading(true);
    const debounceTimer = setTimeout(() => {
      searchMovies(term)
        .then((items) => {
          if (selectedGenre !== 'All') {
            setResults(items.filter((m) => m.genres.includes(selectedGenre)));
          } else {
            setResults(items);
          }
        })
        .catch(console.error)
        .finally(() => setLoading(false));
    }, 250);

    return () => clearTimeout(debounceTimer);
  }, [query, selectedGenre]);

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Header */}
      <div className="mb-8">
        <span className="text-[11px] font-mono tracking-wider text-muted uppercase">
          Catalog Explorer
        </span>
        <h2 className="font-display font-bold text-3xl text-white">
          Browse & Search Movies
        </h2>
        <p className="text-xs text-muted mt-1">
          Explore all 3,416 processed MovieLens catalog titles across genres and eras.
        </p>
      </div>

      {/* Search Input Bar */}
      <div className="relative mb-6">
        <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted" />
        <input
          type="text"
          placeholder="Search by title, director, or theme (e.g., Inception, Noir, Kubrick, Alien)..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="w-full pl-11 pr-4 py-3 rounded-xl bg-surface border border-border focus:border-accent text-white text-sm outline-none transition shadow-sm"
        />
      </div>

      {/* Genre Filter Pills */}
      <div className="flex items-center space-x-2 overflow-x-auto pb-4 mb-8 scrollbar-none">
        {GENRES.map((g) => {
          const active = selectedGenre === g;
          return (
            <button
              key={g}
              onClick={() => setSelectedGenre(g)}
              className={`flex-shrink-0 px-3.5 py-1.5 rounded-lg text-xs font-medium transition ${
                active
                  ? 'bg-accent text-black font-semibold shadow-md shadow-accent/10'
                  : 'bg-surface hover:bg-surface-elevated text-gray-300 border border-border'
              }`}
            >
              {g}
            </button>
          );
        })}
      </div>

      {/* Results Grid */}
      {loading ? (
        <div className="py-24 text-center">
          <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <p className="text-xs text-muted">Searching catalog index...</p>
        </div>
      ) : results.length > 0 ? (
        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4 sm:gap-6">
          {results.map((movie) => (
            <MovieCard
              key={`explore-${movie.movie_idx}`}
              movie={movie}
              onOpenDetail={onOpenDetail}
              onFeedback={onFeedback}
            />
          ))}
        </div>
      ) : (
        <div className="py-20 text-center text-muted border border-dashed border-border rounded-xl">
          <Compass className="w-8 h-8 text-muted mx-auto mb-2" />
          <p className="text-sm text-gray-300">No movies found matching "{query || selectedGenre}".</p>
          <p className="text-xs text-muted mt-1">Try another search keyword or switch genres.</p>
        </div>
      )}
    </div>
  );
};
