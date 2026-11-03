import React, { useRef } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import type { RecommendationItem } from '../types';
import { MovieCard } from './MovieCard';

interface MovieRailProps {
  title: string;
  subtitle?: string;
  movies: RecommendationItem[];
  onOpenDetail: (movieId: number) => void;
  onFeedback: (movieId: number, action: 'like' | 'dislike' | 'save' | 'dismiss') => void;
}

export const MovieRail: React.FC<MovieRailProps> = ({
  title,
  subtitle,
  movies,
  onOpenDetail,
  onFeedback,
}) => {
  const scrollRef = useRef<HTMLDivElement>(null);

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
    }
  };

  return (
    <div className="mb-10">
      {/* Rail Header */}
      <div className="flex items-end justify-between mb-4 px-1">
        <div>
          <h3 className="font-display font-bold text-xl text-white tracking-tight">
            {title}
          </h3>
          {subtitle && (
            <p className="text-xs text-muted mt-0.5">{subtitle}</p>
          )}
        </div>

        {/* Scroll Arrows */}
        <div className="flex items-center space-x-1.5">
          <button
            onClick={() => scroll('left')}
            className="p-1.5 rounded-lg bg-surface hover:bg-surface-elevated border border-border text-muted hover:text-white transition"
            aria-label="Scroll left"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <button
            onClick={() => scroll('right')}
            className="p-1.5 rounded-lg bg-surface hover:bg-surface-elevated border border-border text-muted hover:text-white transition"
            aria-label="Scroll right"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Horizontal Carousel */}
      <div
        ref={scrollRef}
        className="flex space-x-4 overflow-x-auto pb-4 pt-1 px-1 scrollbar-none scroll-smooth"
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
