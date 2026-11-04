import React, { useState, useEffect, useRef } from 'react';
import { Search, X, Star, Sparkles, Film, ArrowRight, CornerDownLeft, Bookmark, ThumbsUp } from 'lucide-react';
import type { RecommendationItem } from '../types';
import { searchMovies } from '../api/client';

interface GlobalSearchBarProps {
  onOpenDetail: (movieId: number) => void;
  onFeedback: (movieId: number, action: 'like' | 'dislike' | 'save' | 'dismiss') => void;
}

const POPULAR_SUGGESTIONS = [
  'The Matrix',
  'Pulp Fiction',
  'Star Wars',
  'Blade Runner',
  'Fargo',
  'Godfather',
  'Toy Story',
  'Inception',
];

const GENRE_SHORTCUTS = [
  'Sci-Fi',
  'Film-Noir',
  'Crime',
  'Animation',
  'Thriller',
  'Drama',
];

export const GlobalSearchBar: React.FC<GlobalSearchBarProps> = ({
  onOpenDetail,
  onFeedback,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<RecommendationItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const modalRef = useRef<HTMLDivElement>(null);

  // Global hotkey: Ctrl+K or Cmd+K
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setIsOpen((prev) => !prev);
      } else if (e.key === 'Escape' && isOpen) {
        setIsOpen(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen]);

  // Focus input when modal opens
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => inputRef.current?.focus(), 50);
      if (!query) {
        // Preload default popular results
        searchMovies('star').then((items) => setResults(items.slice(0, 8)));
      }
    } else {
      setSelectedIndex(0);
    }
  }, [isOpen]);

  // Debounced search
  useEffect(() => {
    if (!query.trim()) {
      return;
    }
    setLoading(true);
    const timer = setTimeout(() => {
      searchMovies(query.trim())
        .then((items) => {
          setResults(items);
          setSelectedIndex(0);
        })
        .catch(console.error)
        .finally(() => setLoading(false));
    }, 200);

    return () => clearTimeout(timer);
  }, [query]);

  // Keyboard navigation within results
  const handleInputKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex((prev) => (prev < results.length - 1 ? prev + 1 : prev));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex((prev) => (prev > 0 ? prev - 1 : 0));
    } else if (e.key === 'Enter') {
      e.preventDefault();
      if (results[selectedIndex]) {
        onOpenDetail(results[selectedIndex].movie_idx);
        setIsOpen(false);
      }
    }
  };

  const handleSelectMovie = (movie: RecommendationItem) => {
    onOpenDetail(movie.movie_idx);
    setIsOpen(false);
  };

  return (
    <>
      {/* Search Input Trigger in Navbar */}
      <div
        onClick={() => setIsOpen(true)}
        className="relative flex items-center cursor-pointer group"
      >
        <div className="flex items-center space-x-2.5 px-3 py-1.5 rounded-xl bg-surface/80 hover:bg-surface border border-border group-hover:border-border-focus transition-all duration-200 w-44 sm:w-64 md:w-72 shadow-inner">
          <Search className="w-3.5 h-3.5 text-muted group-hover:text-accent transition-colors" />
          <span className="text-xs text-muted group-hover:text-gray-300 truncate select-none">
            {query || 'Search movies, genres...'}
          </span>
          <div className="hidden sm:flex items-center ml-auto space-x-1 px-1.5 py-0.5 rounded bg-surface-elevated border border-border text-[10px] font-mono text-muted">
            <span>⌘K</span>
          </div>
        </div>
      </div>

      {/* Spotlight Command / Search Modal */}
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-start justify-center pt-16 sm:pt-24 px-4 bg-black/80 backdrop-blur-md animate-fade-in">
          <div
            ref={modalRef}
            onClick={(e) => e.stopPropagation()}
            className="relative w-full max-w-2xl bg-surface-elevated border border-border/80 rounded-2xl shadow-2xl shadow-black/80 overflow-hidden flex flex-col max-h-[82vh]"
          >
            {/* Search Input Header */}
            <div className="flex items-center px-4 py-3.5 border-b border-border bg-surface/50">
              <Search className="w-5 h-5 text-accent mr-3 flex-shrink-0" />
              <input
                ref={inputRef}
                type="text"
                placeholder="Search across 3,416 MovieLens titles, genres, or eras..."
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={handleInputKeyDown}
                className="w-full bg-transparent text-sm sm:text-base text-white placeholder-muted outline-none font-medium"
              />
              {loading && (
                <div className="w-4 h-4 border-2 border-accent border-t-transparent rounded-full animate-spin mr-2" />
              )}
              {query && (
                <button
                  onClick={() => setQuery('')}
                  className="p-1 rounded-md text-muted hover:text-white mr-1"
                >
                  <X className="w-4 h-4" />
                </button>
              )}
              <button
                onClick={() => setIsOpen(false)}
                className="px-2 py-1 rounded-md bg-surface text-[11px] font-mono text-muted hover:text-white border border-border"
              >
                ESC
              </button>
            </div>

            {/* Quick Suggestion Chips */}
            <div className="px-4 py-2.5 bg-surface/30 border-b border-border/50 flex items-center space-x-2 overflow-x-auto scrollbar-none text-xs">
              <span className="text-[10px] font-mono uppercase tracking-wider text-muted flex-shrink-0">
                Popular:
              </span>
              {POPULAR_SUGGESTIONS.map((item) => (
                <button
                  key={item}
                  onClick={() => setQuery(item)}
                  className="flex-shrink-0 px-2 py-0.5 rounded-full bg-surface hover:bg-surface-hover border border-border text-[11px] text-gray-300 hover:text-white transition"
                >
                  {item}
                </button>
              ))}
              {GENRE_SHORTCUTS.map((genre) => (
                <button
                  key={genre}
                  onClick={() => setQuery(genre)}
                  className="flex-shrink-0 px-2 py-0.5 rounded-full bg-accent/10 hover:bg-accent/20 border border-accent/30 text-[11px] text-accent transition"
                >
                  {genre}
                </button>
              ))}
            </div>

            {/* Results List */}
            <div className="flex-1 overflow-y-auto p-2 divide-y divide-border/30">
              {results.length > 0 ? (
                results.map((movie, index) => {
                  const isSelected = index === selectedIndex;
                  return (
                    <div
                      key={`search-${movie.movie_idx}`}
                      onClick={() => handleSelectMovie(movie)}
                      onMouseEnter={() => setSelectedIndex(index)}
                      className={`flex items-center justify-between p-3 rounded-xl cursor-pointer transition-all ${
                        isSelected
                          ? 'bg-surface border border-accent/40 shadow-md'
                          : 'hover:bg-surface/50 border border-transparent'
                      }`}
                    >
                      <div className="flex items-center space-x-3.5 min-w-0">
                        {/* Film Icon / Poster Thumbnail */}
                        <div className="w-10 h-12 rounded-lg bg-gradient-to-br from-neutral-800 to-slate-900 border border-border flex flex-col items-center justify-center flex-shrink-0 shadow">
                          <Film className="w-4 h-4 text-accent/80" />
                          <span className="text-[9px] font-mono text-muted mt-0.5">
                            {movie.release_year}
                          </span>
                        </div>

                        {/* Title & Metadata */}
                        <div className="min-w-0">
                          <div className="flex items-center space-x-2">
                            <h4 className="text-xs sm:text-sm font-semibold text-white truncate">
                              {movie.title}
                            </h4>
                            {movie.novelty_score > 0.5 && (
                              <span className="hidden sm:inline-flex items-center space-x-1 px-1.5 py-0.2 rounded text-[9px] bg-purple-500/20 text-purple-300 border border-purple-500/30">
                                <Sparkles className="w-2.5 h-2.5" />
                                <span>Deep Cut</span>
                              </span>
                            )}
                          </div>

                          <div className="flex items-center space-x-2 text-[11px] text-muted mt-0.5 truncate">
                            <div className="flex items-center space-x-1 text-amber-400">
                              <Star className="w-3 h-3 fill-amber-400" />
                              <span className="font-semibold text-white">
                                {movie.rating_mean.toFixed(1)}
                              </span>
                            </div>
                            <span>•</span>
                            <span>{movie.genres.slice(0, 3).join(', ')}</span>
                            <span>•</span>
                            <span className="font-mono text-[10px] text-muted-dark">
                              ID #{movie.movieId}
                            </span>
                          </div>
                        </div>
                      </div>

                      {/* Right Quick Action Buttons */}
                      <div className="flex items-center space-x-2 flex-shrink-0 pl-2">
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onFeedback(movie.movie_idx, 'like');
                          }}
                          title="Like"
                          className="p-1.5 rounded-lg bg-surface-elevated hover:bg-surface-hover text-muted hover:text-amber-400 border border-border transition"
                        >
                          <ThumbsUp className="w-3.5 h-3.5" />
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onFeedback(movie.movie_idx, 'save');
                          }}
                          title="Save to Library"
                          className="p-1.5 rounded-lg bg-surface-elevated hover:bg-surface-hover text-muted hover:text-accent border border-border transition"
                        >
                          <Bookmark className="w-3.5 h-3.5" />
                        </button>
                        <div className="hidden sm:flex items-center text-xs text-accent font-medium space-x-1 ml-1">
                          <span>View</span>
                          <ArrowRight className="w-3.5 h-3.5" />
                        </div>
                      </div>
                    </div>
                  );
                })
              ) : !loading && query ? (
                <div className="p-8 text-center text-muted">
                  <p className="text-sm">No movies found matching "{query}".</p>
                  <p className="text-xs text-muted mt-1">Try another keyword, genre, or release year.</p>
                </div>
              ) : (
                <div className="p-8 text-center text-muted">
                  <p className="text-xs font-mono uppercase tracking-wider text-muted-dark">
                    Start typing to search 3,416 movies
                  </p>
                </div>
              )}
            </div>

            {/* Footer Navigation Hints */}
            <div className="px-4 py-2.5 bg-surface/70 border-t border-border flex items-center justify-between text-[11px] text-muted">
              <div className="flex items-center space-x-4">
                <span className="flex items-center space-x-1">
                  <span className="px-1 py-0.5 rounded bg-surface border border-border font-mono text-[10px]">↑</span>
                  <span className="px-1 py-0.5 rounded bg-surface border border-border font-mono text-[10px]">↓</span>
                  <span>Navigate</span>
                </span>
                <span className="flex items-center space-x-1">
                  <CornerDownLeft className="w-3 h-3" />
                  <span>Select</span>
                </span>
                <span className="flex items-center space-x-1">
                  <span className="px-1 py-0.5 rounded bg-surface border border-border font-mono text-[10px]">ESC</span>
                  <span>Close</span>
                </span>
              </div>
              <span className="font-mono text-[10px] text-muted-dark">
                {results.length} matches
              </span>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
