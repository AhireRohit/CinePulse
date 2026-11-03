import React from 'react';
import { Star, ThumbsUp, Bookmark, Sparkles, Sliders, ChevronRight } from 'lucide-react';
import type { RecommendationItem } from '../types';

interface HeroBannerProps {
  movie: RecommendationItem | null;
  onOpenDetail: (movieId: number) => void;
  onFeedback: (movieId: number, action: 'like' | 'dislike' | 'save' | 'dismiss') => void;
}

export const HeroBanner: React.FC<HeroBannerProps> = ({
  movie,
  onOpenDetail,
  onFeedback,
}) => {
  if (!movie) {
    return null;
  }

  const matchPercentage = Math.round(movie.relevance_score * 100);

  return (
    <div className="relative w-full rounded-2xl overflow-hidden bg-gradient-to-r from-surface-elevated via-surface to-background border border-border p-8 md:p-12 mb-10 shadow-2xl">
      {/* Background ambient lighting */}
      <div className="absolute top-0 right-0 -mr-20 -mt-20 w-96 h-96 rounded-full bg-amber-500/5 blur-3xl pointer-events-none" />

      <div className="relative z-10 max-w-3xl">
        {/* Top Feature Tag */}
        <div className="flex flex-wrap items-center gap-3 mb-4">
          <span className="px-3 py-1 rounded-full bg-amber-500/10 border border-amber-500/30 text-xs font-semibold text-accent uppercase tracking-wider flex items-center space-x-1.5">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Top Recommendation For You</span>
          </span>

          <span className="text-xs text-muted font-mono">
            {movie.reason}
          </span>
        </div>

        {/* Title */}
        <h1 className="font-display font-extrabold text-3xl sm:text-5xl text-white tracking-tight leading-tight mb-4">
          {movie.title}
        </h1>

        {/* Metadata Row */}
        <div className="flex flex-wrap items-center gap-4 text-xs text-muted mb-6">
          <div className="flex items-center space-x-1 text-white font-medium">
            <Star className="w-4 h-4 text-amber-400 fill-amber-400" />
            <span>{movie.rating_mean.toFixed(1)} / 5.0</span>
          </div>

          <span className="w-1 h-1 rounded-full bg-muted" />
          <span>{movie.release_year}</span>

          <span className="w-1 h-1 rounded-full bg-muted" />
          <div className="flex items-center space-x-1.5">
            {movie.genres.map((g) => (
              <span key={g} className="px-2 py-0.5 rounded bg-surface border border-border text-gray-300">
                {g}
              </span>
            ))}
          </div>

          <span className="w-1 h-1 rounded-full bg-muted" />
          <span className="px-2 py-0.5 rounded bg-green-500/10 border border-green-500/20 text-green-400 font-semibold font-mono">
            {matchPercentage}% Match
          </span>
        </div>

        {/* Narrative / Context */}
        <p className="text-sm text-gray-300 leading-relaxed max-w-2xl mb-8">
          Selected by the multi-stage neural retrieval and ranking pipeline. Combines Two-Tower deep embedding alignment with your historical preference patterns, balanced for genre diversity via Maximal Marginal Relevance.
        </p>

        {/* Action Buttons */}
        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={() => onFeedback(movie.movie_idx, 'like')}
            className="flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-accent text-black font-semibold text-xs hover:bg-accent-hover transition shadow-lg shadow-accent/10"
          >
            <ThumbsUp className="w-4 h-4" />
            <span>Like This Match</span>
          </button>

          <button
            onClick={() => onFeedback(movie.movie_idx, 'save')}
            className="flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-surface-elevated hover:bg-surface-hover border border-border text-white text-xs font-medium transition"
          >
            <Bookmark className="w-4 h-4" />
            <span>Save to Library</span>
          </button>

          <button
            onClick={() => onOpenDetail(movie.movie_idx)}
            className="flex items-center space-x-1.5 px-4 py-2.5 rounded-xl bg-transparent hover:bg-surface border border-transparent hover:border-border text-muted hover:text-white text-xs font-medium transition"
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>Recommendation Diagnostics</span>
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
};
