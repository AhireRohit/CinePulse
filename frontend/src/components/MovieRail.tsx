import React, { useRef, useState, useEffect } from 'react';
import { ChevronLeft, ChevronRight, Film } from 'lucide-react';
import type { RecommendationItem } from '../types';
import { MovieCard } from './MovieCard';

interface MovieRailProps {
  title: string;
  subtitle?: string;
  movies: RecommendationItem[];
  onOpenDetail: (movieId: number) => void;
  onFeedback: (
    movieId: number,
    action: 'like' | 'dislike' | 'save' | 'dismiss' | 'rate',
    rating?: number
  ) => void;
}

export const MovieRail: React.FC<MovieRailProps> = ({
  title,
  subtitle,
  movies,
  onOpenDetail,
  onFeedback,
}) => {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(true);

  const checkScroll = () => {
    if (scrollRef.current) {
      const { scrollLeft, scrollWidth, clientWidth } = scrollRef.current;
      setCanScrollLeft(scrollLeft > 20);
      setCanScrollRight(scrollLeft + clientWidth < scrollWidth - 20);
    }
  };

  useEffect(() => {
    checkScroll();
  }, [movies]);

  if (!movies || movies.length === 0) {
    return null;
  }

  const scroll = (direction: 'left' | 'right') => {
    if (scrollRef.current) {
      const { scrollLeft, clientWidth } = scrollRef.current;
      const scrollAmount = clientWidth * 0.75;
      scrollRef.current.scrollTo({
        left: direction === 'left' ? scrollLeft - scrollAmount : scrollLeft + scrollAmount,
        behavior: 'smooth',
      });
      setTimeout(checkScroll, 350);
    }
  };

  return (
    <div className="mb-12 relative group/rail">
      {/* Rail Header */}
      <div className="flex items-end justify-between mb-4 px-1">
        <div>
          <div className="flex items-center space-x-2">
            <Film className="w-4 h-4 text-accent" />
            <h3 className="font-display font-bold text-xl sm:text-2xl text-white tracking-tight">
              {title}
            </h3>
            <span className="px-2 py-0.5 rounded-full bg-surface-elevated border border-border text-[10px] font-mono text-muted">
              {movies.length} titles
            </span>
          </div>
          {subtitle && (
            <p className="text-xs text-muted mt-1 max-w-xl">{subtitle}</p>
          )}
        </div>

        {/* Scroll Controls */}
        <div className="flex items-center space-x-2">
          <button
            onClick={() => scroll('left')}
            disabled={!canScrollLeft}
            className="p-2 rounded-xl bg-surface hover:bg-surface-elevated border border-border text-muted hover:text-white transition disabled:opacity-30 disabled:cursor-not-allowed shadow-sm"
            aria-label="Scroll left"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <button
            onClick={() => scroll('right')}
            disabled={!canScrollRight}
            className="p-2 rounded-xl bg-surface hover:bg-surface-elevated border border-border text-muted hover:text-white transition disabled:opacity-30 disabled:cursor-not-allowed shadow-sm"
            aria-label="Scroll right"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Horizontal Carousel */}
      <div
        ref={scrollRef}
        onScroll={checkScroll}
        className="flex space-x-4 sm:space-x-5 overflow-x-auto pb-4 pt-1 px-1 scrollbar-none scroll-smooth"
        style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
      >
        {movies.map((movie) => (
          <MovieCard
            key={`${title}-${movie.movie_idx}`}
            movie={movie}
            onOpenDetail={onOpenDetail}
            onFeedback={onFeedback}
          />
        ))}
      </div>
    </div>
  );
};
