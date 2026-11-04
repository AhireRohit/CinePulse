import React, { useState, useEffect, useMemo } from 'react';
import { Search, Compass, X, ArrowUpDown, LayoutGrid, List } from 'lucide-react';
import type { RecommendationItem } from '../types';
import { searchMovies } from '../api/client';
import { MovieCard } from '../components/MovieCard';

interface ExplorePageProps {
  onOpenDetail: (movieId: number) => void;
  onFeedback: (
    movieId: number,
    action: 'like' | 'dislike' | 'save' | 'dismiss' | 'rate',
    rating?: number
  ) => void;
}

const GENRES = [
  'All', 'Sci-Fi', 'Crime', 'Drama', 'Film-Noir', 'Mystery', 'Animation',
  'Action', 'Thriller', 'Comedy', 'Adventure', 'Romance', 'Horror', 'War'
];

const ERAS = [
  { id: 'all', label: 'All Eras' },
  { id: '90s', label: '1990s (Peak 90s)', min: 1990, max: 1999 },
  { id: '80s', label: '1980s Classics', min: 1980, max: 1989 },
  { id: '70s', label: '1970s New Hollywood', min: 1970, max: 1979 },
  { id: 'golden', label: 'Golden Age (Pre-1970)', min: 1900, max: 1969 },
];

export const ExplorePage: React.FC<ExplorePageProps> = ({
  onOpenDetail,
  onFeedback,
}) => {
  const [query, setQuery] = useState('');
  const [selectedGenre, setSelectedGenre] = useState('All');
  const [selectedEra, setSelectedEra] = useState('all');
  const [sortBy, setSortBy] = useState<'rating' | 'year_desc' | 'year_asc' | 'popularity'>('rating');
  const [viewMode, setViewMode] = useState<'grid' | 'compact'>('grid');
  const [rawResults, setRawResults] = useState<RecommendationItem[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const term = query.trim() || (selectedGenre === 'All' ? 'the' : selectedGenre);
    setLoading(true);
    const debounceTimer = setTimeout(() => {
      searchMovies(term)
        .then((items) => {
          setRawResults(items);
        })
        .catch(console.error)
        .finally(() => setLoading(false));
    }, 200);

    return () => clearTimeout(debounceTimer);
  }, [query, selectedGenre]);

  // Client-side filtering & sorting
  const filteredAndSortedResults = useMemo(() => {
    let list = [...rawResults];

    // Filter by genre
    if (selectedGenre !== 'All') {
      list = list.filter((m) => m.genres.includes(selectedGenre));
    }

    // Filter by era
    if (selectedEra !== 'all') {
      const era = ERAS.find((e) => e.id === selectedEra);
      if (era && era.min && era.max) {
        list = list.filter((m) => m.release_year >= era.min! && m.release_year <= era.max!);
      }
    }

    // Sort
    list.sort((a, b) => {
      if (sortBy === 'rating') {
        return b.rating_mean - a.rating_mean;
      } else if (sortBy === 'year_desc') {
        return b.release_year - a.release_year;
      } else if (sortBy === 'year_asc') {
        return a.release_year - b.release_year;
      } else if (sortBy === 'popularity') {
        return b.rating_count - a.rating_count;
      }
      return 0;
    });

    return list;
  }, [rawResults, selectedGenre, selectedEra, sortBy]);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
      {/* Header */}
      <div className="mb-6">
        <div className="flex items-center space-x-2 text-[11px] font-mono tracking-wider text-muted uppercase">
          <Compass className="w-3.5 h-3.5 text-accent" />
          <span>Interactive Catalog Explorer</span>
        </div>
        <h2 className="font-display font-extrabold text-3xl sm:text-4xl text-white tracking-tight mt-1">
          Explore MovieLens Catalog
        </h2>
        <p className="text-xs sm:text-sm text-muted mt-1 max-w-2xl">
          Search, filter by cinematic eras, and explore all 3,416 processed feature films with real-time feedback.
        </p>
      </div>

      {/* Main Search Bar & Quick Tools */}
      <div className="flex flex-col sm:flex-row items-center gap-3 mb-6">
        <div className="relative flex-1 w-full">
          <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted" />
          <input
            type="text"
            placeholder="Search by title, director, theme, or year (e.g., Matrix, Noir, 1994, Kubrick)..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="w-full pl-11 pr-10 py-3 rounded-xl bg-surface border border-border focus:border-accent text-white text-sm outline-none transition shadow-inner font-medium placeholder-muted"
          />
          {query && (
            <button
              onClick={() => setQuery('')}
              className="absolute right-3.5 top-1/2 transform -translate-y-1/2 p-1 rounded-md text-muted hover:text-white"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Sort Dropdown */}
        <div className="flex items-center space-x-2 w-full sm:w-auto justify-between sm:justify-start">
          <div className="flex items-center space-x-1.5 px-3 py-2.5 rounded-xl bg-surface border border-border text-xs text-gray-200">
            <ArrowUpDown className="w-3.5 h-3.5 text-accent" />
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="bg-transparent text-xs text-white outline-none cursor-pointer"
            >
              <option value="rating" className="bg-surface-elevated">Highest Rated</option>
              <option value="popularity" className="bg-surface-elevated">Most Reviewed</option>
              <option value="year_desc" className="bg-surface-elevated">Newest Release</option>
              <option value="year_asc" className="bg-surface-elevated">Oldest Classic</option>
            </select>
          </div>

          {/* View Mode Toggle */}
          <div className="flex items-center p-1 rounded-xl bg-surface border border-border">
            <button
              onClick={() => setViewMode('grid')}
              className={`p-1.5 rounded-lg transition ${
                viewMode === 'grid' ? 'bg-surface-elevated text-accent shadow-sm' : 'text-muted hover:text-white'
              }`}
              title="Grid View"
            >
              <LayoutGrid className="w-4 h-4" />
            </button>
            <button
              onClick={() => setViewMode('compact')}
              className={`p-1.5 rounded-lg transition ${
                viewMode === 'compact' ? 'bg-surface-elevated text-accent shadow-sm' : 'text-muted hover:text-white'
              }`}
              title="List View"
            >
              <List className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Era Selector Pills */}
      <div className="flex items-center space-x-2 overflow-x-auto pb-2 mb-4 scrollbar-none text-xs">
        <span className="text-[10px] font-mono uppercase tracking-wider text-muted flex-shrink-0 mr-1">
          Era:
        </span>
        {ERAS.map((era) => {
          const active = selectedEra === era.id;
          return (
            <button
              key={era.id}
              onClick={() => setSelectedEra(era.id)}
              className={`flex-shrink-0 px-3 py-1 rounded-lg text-xs font-medium transition ${
                active
                  ? 'bg-accent/20 text-accent border border-accent/40 font-semibold'
                  : 'bg-surface hover:bg-surface-elevated text-muted hover:text-white border border-border'
              }`}
            >
              {era.label}
            </button>
          );
        })}
      </div>

      {/* Genre Filter Pills */}
      <div className="flex items-center space-x-2 overflow-x-auto pb-3 mb-6 scrollbar-none">
        {GENRES.map((g) => {
          const active = selectedGenre === g;
          return (
            <button
              key={g}
              onClick={() => setSelectedGenre(g)}
              className={`flex-shrink-0 px-3.5 py-1.5 rounded-xl text-xs font-semibold transition ${
                active
                  ? 'bg-accent text-black font-bold shadow-md shadow-accent/20'
                  : 'bg-surface hover:bg-surface-elevated text-gray-300 border border-border'
              }`}
            >
              {g}
            </button>
          );
        })}
      </div>

      {/* Search Result Counter & Info */}
      <div className="flex items-center justify-between text-xs text-muted mb-6 pb-2 border-b border-border/60 font-mono">
        <span>
          Showing <strong className="text-white">{filteredAndSortedResults.length}</strong> matching titles
        </span>
        <span>
          Filter: {selectedGenre} • Era: {ERAS.find((e) => e.id === selectedEra)?.label}
        </span>
      </div>

      {/* Results Rendering */}
      {loading ? (
        <div className="py-24 text-center">
          <div className="w-10 h-10 border-2 border-accent border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <p className="text-xs text-muted font-mono">Searching candidate index...</p>
        </div>
      ) : filteredAndSortedResults.length > 0 ? (
        viewMode === 'grid' ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4 sm:gap-6">
            {filteredAndSortedResults.map((movie) => (
              <MovieCard
                key={`explore-${movie.movie_idx}`}
                movie={movie}
                onOpenDetail={onOpenDetail}
                onFeedback={onFeedback}
              />
            ))}
          </div>
        ) : (
          <div className="divide-y divide-border/40 bg-surface rounded-2xl border border-border overflow-hidden">
            {filteredAndSortedResults.map((movie) => (
              <div
                key={`list-${movie.movie_idx}`}
                onClick={() => onOpenDetail(movie.movie_idx)}
                className="p-4 flex items-center justify-between hover:bg-surface-elevated cursor-pointer transition"
              >
                <div className="min-w-0 pr-4">
                  <div className="flex items-center space-x-2">
                    <h4 className="text-sm font-semibold text-white truncate">
                      {movie.title}
                    </h4>
                    <span className="text-xs font-mono text-muted">
                      ({movie.release_year})
                    </span>
                  </div>
                  <div className="flex items-center space-x-2 text-xs text-muted mt-1">
                    <span>{movie.genres.join(', ')}</span>
                    <span>•</span>
                    <span className="text-amber-400 font-semibold">{movie.rating_mean.toFixed(1)}★</span>
                    <span>•</span>
                    <span>{movie.rating_count} reviews</span>
                  </div>
                </div>

                <div className="flex items-center space-x-2 flex-shrink-0">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onFeedback(movie.movie_idx, 'like');
                    }}
                    className="px-3 py-1.5 rounded-lg bg-surface border border-border hover:border-accent text-xs text-gray-200 transition"
                  >
                    Like
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onFeedback(movie.movie_idx, 'save');
                    }}
                    className="px-3 py-1.5 rounded-lg bg-accent/20 border border-accent/40 text-accent text-xs font-medium transition"
                  >
                    Save
                  </button>
                </div>
              </div>
            ))}
          </div>
        )
      ) : (
        <div className="py-20 text-center text-muted border border-dashed border-border rounded-2xl bg-surface/30">
          <Compass className="w-10 h-10 text-muted mx-auto mb-3" />
          <p className="text-base text-gray-200 font-semibold">No movies found</p>
          <p className="text-xs text-muted mt-1">Try another keyword, switch eras, or select 'All' genres.</p>
        </div>
      )}
    </div>
  );
};
