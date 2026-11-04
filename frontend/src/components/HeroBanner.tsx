import React, { useState } from 'react';
import { Star, ThumbsUp, Bookmark, Sparkles, Sliders, ChevronRight, Play, EyeOff } from 'lucide-react';
import type { RecommendationItem } from '../types';

interface HeroBannerProps {
  movie: RecommendationItem | null;
  onOpenDetail: (movieId: number) => void;
  onFeedback: (
    movieId: number,
    action: 'like' | 'dislike' | 'save' | 'dismiss' | 'rate',
    rating?: number
  ) => void;
}

export const HeroBanner: React.FC<HeroBannerProps> = ({
  movie,
  onOpenDetail,
  onFeedback,
}) => {
  const [liked, setLiked] = useState(false);
  const [saved, setSaved] = useState(false);
  const [userRating, setUserRating] = useState<number | null>(null);

  if (!movie) {
    return null;
  }

  const matchPercentage = Math.round(movie.relevance_score * 100);

  const handleLike = () => {
    const next = !liked;
    setLiked(next);
    onFeedback(movie.movie_idx, 'like');
  };

  const handleSave = () => {
    const next = !saved;
    setSaved(next);
    onFeedback(movie.movie_idx, 'save');
  };

  const handleRate = (star: number) => {
    setUserRating(star);
    onFeedback(movie.movie_idx, 'rate', star);
  };

  return (
    <div className="relative w-full rounded-2xl overflow-hidden bg-gradient-to-r from-surface-elevated via-surface to-background border border-border p-6 sm:p-10 md:p-12 mb-10 shadow-2xl group">
      {/* Background ambient lighting effects */}
      <div className="absolute top-0 right-0 -mr-24 -mt-24 w-[420px] h-[420px] rounded-full bg-amber-500/10 blur-[100px] pointer-events-none" />
      <div className="absolute bottom-0 left-1/3 -mb-20 w-80 h-80 rounded-full bg-indigo-500/10 blur-[80px] pointer-events-none" />

      {/* Decorative film frame accent */}
      <div className="absolute top-4 right-6 hidden md:flex items-center space-x-2 text-xs font-mono text-muted/60">
        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
        <span>NEURAL RETRIEVAL RANK #1</span>
      </div>

      <div className="relative z-10 max-w-3xl">
        {/* Top Feature Tag & Reason */}
        <div className="flex flex-wrap items-center gap-3 mb-4">
          <span className="px-3 py-1 rounded-full bg-gradient-to-r from-amber-500/20 to-amber-600/10 border border-amber-500/40 text-xs font-semibold text-accent uppercase tracking-wider flex items-center space-x-1.5 shadow-sm">
            <Sparkles className="w-3.5 h-3.5 text-amber-400" />
            <span>Top Spotlight Recommendation</span>
          </span>

          <span className="text-xs text-muted font-mono flex items-center space-x-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-accent" />
            <span>{movie.reason || 'Optimal multi-stage alignment'}</span>
          </span>
        </div>

        {/* Title */}
        <h1 className="font-display font-extrabold text-3xl sm:text-5xl md:text-6xl text-white tracking-tight leading-none mb-3">
          {movie.title}
        </h1>

        {/* Metadata Row */}
        <div className="flex flex-wrap items-center gap-3 sm:gap-4 text-xs text-muted mb-5">
          <div className="flex items-center space-x-1.5 px-2.5 py-1 rounded-lg bg-black/60 border border-white/10 text-white font-semibold shadow-sm">
            <Star className="w-4 h-4 text-amber-400 fill-amber-400" />
            <span>{movie.rating_mean.toFixed(1)}</span>
            <span className="text-[10px] text-muted font-normal">/ 5.0</span>
          </div>

          <span className="font-mono text-gray-300 font-medium">{movie.release_year}</span>
          <span className="w-1 h-1 rounded-full bg-muted" />

          <div className="flex items-center space-x-1.5">
            {movie.genres.map((g) => (
              <span
                key={g}
                className="px-2.5 py-0.5 rounded-md bg-surface-elevated border border-border text-gray-200 text-xs font-medium"
              >
                {g}
              </span>
            ))}
          </div>

          <span className="w-1 h-1 rounded-full bg-muted" />
          <span className="px-2.5 py-0.5 rounded-md bg-emerald-500/15 border border-emerald-500/30 text-emerald-400 font-bold font-mono text-xs flex items-center space-x-1">
            <span>{matchPercentage}% Match</span>
          </span>
        </div>

        {/* Narrative / Context */}
        <p className="text-xs sm:text-sm text-gray-300 leading-relaxed max-w-2xl mb-6">
          Selected by the multi-stage neural retrieval and ranking pipeline. Combines Two-Tower deep embedding alignment with your historical preference patterns, balanced for genre diversity via Maximal Marginal Relevance.
        </p>

        {/* Interactive 5-Star Rating on Hero */}
        <div className="flex items-center space-x-3 mb-7 p-3 rounded-xl bg-surface/60 border border-border/80 w-fit">
          <span className="text-xs text-muted font-medium">Your Rating:</span>
          <div className="flex items-center space-x-1">
            {[1, 2, 3, 4, 5].map((star) => (
              <button
                key={star}
                onClick={() => handleRate(star)}
                className="p-1 hover:scale-125 transition-transform"
                title={`Rate ${star} star${star > 1 ? 's' : ''}`}
              >
                <Star
                  className={`w-4 h-4 sm:w-5 sm:h-5 transition-colors ${
                    (userRating || 0) >= star
                      ? 'text-amber-400 fill-amber-400'
                      : 'text-muted-dark hover:text-amber-400'
                  }`}
                />
              </button>
            ))}
          </div>
          {userRating && (
            <span className="text-xs text-accent font-mono font-semibold ml-1">
              Rated {userRating}★ (Session Updated)
            </span>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={() => onOpenDetail(movie.movie_idx)}
            className="flex items-center space-x-2 px-5 py-2.5 rounded-xl bg-accent text-black font-bold text-xs sm:text-sm hover:bg-accent-hover transition shadow-lg shadow-accent/20 hover:scale-[1.02]"
          >
            <Play className="w-4 h-4 fill-black" />
            <span>Inspect Film & Signals</span>
          </button>

          <button
            onClick={handleLike}
            className={`flex items-center space-x-2 px-4 py-2.5 rounded-xl text-xs sm:text-sm font-semibold transition border ${
              liked
                ? 'bg-amber-500/20 text-amber-400 border-amber-500/40'
                : 'bg-surface-elevated hover:bg-surface-hover border-border text-white'
            }`}
          >
            <ThumbsUp className={`w-4 h-4 ${liked ? 'fill-amber-400' : ''}`} />
            <span>{liked ? 'Liked' : 'Like'}</span>
          </button>

          <button
            onClick={handleSave}
            className={`flex items-center space-x-2 px-4 py-2.5 rounded-xl text-xs sm:text-sm font-semibold transition border ${
              saved
                ? 'bg-accent/20 text-accent border-accent/40'
                : 'bg-surface-elevated hover:bg-surface-hover border-border text-white'
            }`}
          >
            <Bookmark className={`w-4 h-4 ${saved ? 'fill-accent' : ''}`} />
            <span>{saved ? 'Saved' : 'Watchlist'}</span>
          </button>

          <button
            onClick={() => onFeedback(movie.movie_idx, 'dismiss')}
            className="flex items-center space-x-1.5 px-3 py-2.5 rounded-xl bg-surface-elevated hover:bg-red-500/20 border border-border text-muted hover:text-red-400 text-xs font-medium transition"
            title="Dismiss from recommendations"
          >
            <EyeOff className="w-4 h-4" />
          </button>

          <button
            onClick={() => onOpenDetail(movie.movie_idx)}
            className="flex items-center space-x-1.5 px-3.5 py-2.5 rounded-xl bg-transparent hover:bg-surface border border-transparent hover:border-border text-muted hover:text-white text-xs font-medium transition ml-auto hidden sm:flex"
          >
            <Sliders className="w-3.5 h-3.5 text-accent" />
            <span>Model Diagnostics</span>
            <ChevronRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
};
