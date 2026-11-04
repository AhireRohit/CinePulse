import React, { useState } from 'react';
import { Star, ThumbsUp, ThumbsDown, Bookmark, EyeOff, Info, Sparkles, Film, Play } from 'lucide-react';
import type { RecommendationItem } from '../types';

interface MovieCardProps {
  movie: RecommendationItem;
  onOpenDetail: (movieId: number) => void;
  onFeedback: (
    movieId: number,
    action: 'like' | 'dislike' | 'save' | 'dismiss' | 'rate',
    rating?: number
  ) => void;
  isSaved?: boolean;
  isLiked?: boolean;
}

// Genre-specific artistic color themes & styling
const GENRE_THEMES: Record<string, { gradient: string; glow: string; accentColor: string; watermark: string }> = {
  'Sci-Fi': {
    gradient: 'from-cyan-950 via-slate-900 to-indigo-950',
    glow: 'group-hover:shadow-cyan-500/20',
    accentColor: '#38BDF8',
    watermark: 'SCI-FI',
  },
  'Film-Noir': {
    gradient: 'from-zinc-950 via-stone-900 to-neutral-950',
    glow: 'group-hover:shadow-amber-500/10',
    accentColor: '#E2E8F0',
    watermark: 'NOIR',
  },
  'Crime': {
    gradient: 'from-stone-950 via-neutral-900 to-red-950/40',
    glow: 'group-hover:shadow-red-500/15',
    accentColor: '#F87171',
    watermark: 'CRIME',
  },
  'Action': {
    gradient: 'from-amber-950/70 via-stone-900 to-zinc-950',
    glow: 'group-hover:shadow-amber-500/20',
    accentColor: '#FBBF24',
    watermark: 'ACTION',
  },
  'Animation': {
    gradient: 'from-purple-950 via-slate-900 to-blue-950',
    glow: 'group-hover:shadow-purple-500/20',
    accentColor: '#C084FC',
    watermark: 'ANIMATION',
  },
  'Drama': {
    gradient: 'from-slate-950 via-stone-900 to-neutral-950',
    glow: 'group-hover:shadow-amber-500/10',
    accentColor: '#F59E0B',
    watermark: 'CINEMA',
  },
  'Horror': {
    gradient: 'from-red-950/80 via-neutral-950 to-stone-950',
    glow: 'group-hover:shadow-red-600/20',
    accentColor: '#EF4444',
    watermark: 'HORROR',
  },
  'Thriller': {
    gradient: 'from-teal-950 via-neutral-900 to-slate-950',
    glow: 'group-hover:shadow-teal-500/20',
    accentColor: '#2DD4BF',
    watermark: 'THRILLER',
  },
  'Adventure': {
    gradient: 'from-emerald-950 via-stone-900 to-zinc-950',
    glow: 'group-hover:shadow-emerald-500/20',
    accentColor: '#34D399',
    watermark: 'EPIC',
  },
  'Comedy': {
    gradient: 'from-blue-950 via-indigo-950 to-neutral-950',
    glow: 'group-hover:shadow-blue-500/20',
    accentColor: '#60A5FA',
    watermark: 'COMEDY',
  },
};

export const MovieCard: React.FC<MovieCardProps> = ({
  movie,
  onOpenDetail,
  onFeedback,
  isSaved = false,
  isLiked = false,
}) => {
  const [liked, setLiked] = useState(isLiked);
  const [saved, setSaved] = useState(isSaved);
  const [dismissed, setDismissed] = useState(false);
  const [hoverRating, setHoverRating] = useState<number | null>(null);
  const [currentRating, setCurrentRating] = useState<number | null>(null);

  if (dismissed) {
    return null;
  }

  const primaryGenre = movie.genres[0] || 'Drama';
  const theme = GENRE_THEMES[primaryGenre] || GENRE_THEMES['Drama'];

  const handleAction = (
    e: React.MouseEvent,
    action: 'like' | 'dislike' | 'save' | 'dismiss'
  ) => {
    e.stopPropagation();
    if (action === 'like') {
      const next = !liked;
      setLiked(next);
      onFeedback(movie.movie_idx, 'like');
    } else if (action === 'save') {
      const next = !saved;
      setSaved(next);
      onFeedback(movie.movie_idx, 'save');
    } else if (action === 'dismiss' || action === 'dislike') {
      setDismissed(true);
      onFeedback(movie.movie_idx, action);
    }
  };

  const handleRate = (e: React.MouseEvent, star: number) => {
    e.stopPropagation();
    setCurrentRating(star);
    onFeedback(movie.movie_idx, 'rate', star);
  };

  return (
    <div
      onClick={() => onOpenDetail(movie.movie_idx)}
      className={`group relative flex-shrink-0 w-48 sm:w-56 cursor-pointer select-none rounded-xl bg-surface border border-border hover:border-accent/50 hover:shadow-2xl ${theme.glow} transition-all duration-300 transform hover:-translate-y-1.5 overflow-hidden flex flex-col`}
    >
      {/* Poster / Typographic Art Cover */}
      <div className={`relative h-64 sm:h-72 w-full bg-gradient-to-br ${theme.gradient} p-4 flex flex-col justify-between border-b border-border/40 overflow-hidden`}>
        {/* Subtle Watermark Genre Typography in Background */}
        <div className="absolute -right-4 -bottom-3 text-[52px] sm:text-[60px] font-black tracking-tighter text-white/[0.04] pointer-events-none select-none font-display leading-none">
          {theme.watermark}
        </div>

        {/* Ambient Top Glow Line */}
        <div className="absolute top-0 inset-x-0 h-px bg-gradient-to-r from-transparent via-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />

        {/* Top Badges */}
        <div className="flex items-center justify-between z-10">
          {/* Rating Pill */}
          <div className="flex items-center space-x-1 px-2.5 py-0.5 rounded-full bg-black/70 backdrop-blur-md border border-white/10 text-[11px] font-semibold text-white shadow-sm">
            <Star className="w-3 h-3 text-amber-400 fill-amber-400" />
            <span>{movie.rating_mean.toFixed(1)}</span>
          </div>

          {/* Deep Cut / Novelty Sparkle Badge */}
          {movie.novelty_score > 0.5 ? (
            <div className="flex items-center space-x-1 px-2 py-0.5 rounded-full bg-purple-500/20 backdrop-blur-md border border-purple-500/40 text-[10px] font-medium text-purple-300 shadow">
              <Sparkles className="w-2.5 h-2.5" />
              <span>Deep Cut</span>
            </div>
          ) : (
            <div className="flex items-center space-x-1 px-2 py-0.5 rounded-full bg-black/50 backdrop-blur-sm border border-white/10 text-[10px] font-mono text-gray-300">
              <span>{movie.release_year}</span>
            </div>
          )}
        </div>

        {/* Center Title Display */}
        <div className="my-auto z-10 text-center px-1">
          <div className="w-8 h-8 rounded-full bg-black/40 border border-white/10 flex items-center justify-center mx-auto mb-2 text-white/50 group-hover:text-accent group-hover:border-accent/40 group-hover:scale-110 transition-all">
            <Film className="w-4 h-4" />
          </div>
          <h4 className="font-display font-bold text-base sm:text-lg text-gray-100 tracking-tight leading-snug line-clamp-3 group-hover:text-accent transition-colors">
            {movie.title}
          </h4>
          <span className="block mt-1 font-mono text-[11px] text-muted">
            {movie.release_year} • {primaryGenre}
          </span>
        </div>

        {/* Bottom Recommendation Reason Pill */}
        <div className="z-10">
          <div className="px-2 py-1 rounded-lg bg-black/70 backdrop-blur-md border border-white/10 text-[10px] text-gray-300 font-medium truncate flex items-center space-x-1.5 shadow">
            <span className="w-1.5 h-1.5 rounded-full bg-accent flex-shrink-0" />
            <span className="truncate">{movie.reason || 'Curated match'}</span>
          </div>
        </div>

        {/* Action Overlay on Hover */}
        <div className="absolute inset-0 bg-black/85 backdrop-blur-md opacity-0 group-hover:opacity-100 transition-all duration-200 flex flex-col justify-between p-4 z-20">
          {/* Top Dismiss Button */}
          <div className="flex justify-between items-center">
            <span className="text-[10px] font-mono text-accent uppercase tracking-wider">
              Quick React
            </span>
            <button
              onClick={(e) => handleAction(e, 'dismiss')}
              title="Dismiss / Hide this movie"
              className="p-1.5 rounded-full bg-surface-elevated hover:bg-red-500/20 text-muted hover:text-red-400 border border-border transition"
            >
              <EyeOff className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Center Details & Synopsis Preview */}
          <div className="text-center px-1 my-auto">
            <p className="text-xs text-gray-200 line-clamp-2 mb-2 font-medium">
              {movie.full_title}
            </p>
            <div className="flex flex-wrap justify-center gap-1 mb-3">
              {movie.genres.slice(0, 3).map((g) => (
                <span
                  key={g}
                  className="px-1.5 py-0.5 rounded text-[10px] bg-surface-elevated text-gray-300 border border-border"
                >
                  {g}
                </span>
              ))}
            </div>

            {/* Interactive 5-Star Rating directly on card! */}
            <div className="bg-surface/80 p-2 rounded-xl border border-border inline-block">
              <span className="block text-[9px] font-mono text-muted uppercase tracking-wider mb-1">
                {currentRating ? `Rated ${currentRating}★` : 'Click to Rate'}
              </span>
              <div className="flex items-center justify-center space-x-1">
                {[1, 2, 3, 4, 5].map((star) => (
                  <button
                    key={star}
                    onClick={(e) => handleRate(e, star)}
                    onMouseEnter={() => setHoverRating(star)}
                    onMouseLeave={() => setHoverRating(null)}
                    title={`Rate ${star} star${star > 1 ? 's' : ''}`}
                    className="p-0.5 hover:scale-125 transition-transform"
                  >
                    <Star
                      className={`w-4 h-4 transition-colors ${
                        (hoverRating || currentRating || 0) >= star
                          ? 'text-amber-400 fill-amber-400'
                          : 'text-muted-dark hover:text-amber-400'
                      }`}
                    />
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Bottom Action Buttons */}
          <div className="flex items-center justify-around pt-2 border-t border-border/80">
            <button
              onClick={(e) => handleAction(e, 'like')}
              title={liked ? 'Liked' : 'Like'}
              className={`p-2 rounded-xl transition ${
                liked
                  ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40 shadow-sm'
                  : 'bg-surface-elevated hover:bg-surface-hover text-gray-300 border border-border'
              }`}
            >
              <ThumbsUp className={`w-4 h-4 ${liked ? 'fill-amber-400' : ''}`} />
            </button>

            <button
              onClick={(e) => handleAction(e, 'dislike')}
              title="Dislike & Suppress"
              className="p-2 rounded-xl bg-surface-elevated hover:bg-red-500/20 text-gray-300 hover:text-red-400 border border-border transition"
            >
              <ThumbsDown className="w-4 h-4" />
            </button>

            <button
              onClick={(e) => handleAction(e, 'save')}
              title={saved ? 'Saved in Library' : 'Save to Library'}
              className={`p-2 rounded-xl transition ${
                saved
                  ? 'bg-accent/20 text-accent border border-accent/40 shadow-sm'
                  : 'bg-surface-elevated hover:bg-surface-hover text-gray-300 border border-border'
              }`}
            >
              <Bookmark className={`w-4 h-4 ${saved ? 'fill-accent' : ''}`} />
            </button>

            <button
              onClick={(e) => {
                e.stopPropagation();
                onOpenDetail(movie.movie_idx);
              }}
              title="Inspect Details & Signals"
              className="p-2 rounded-xl bg-surface-elevated hover:bg-surface-hover text-gray-300 hover:text-white border border-border transition"
            >
              <Info className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Card Info Below */}
      <div className="p-3 bg-surface flex-1 flex flex-col justify-between">
        <div>
          <h5 className="font-semibold text-xs text-gray-200 truncate group-hover:text-accent transition-colors">
            {movie.title}
          </h5>
          <div className="mt-1 flex items-center justify-between text-[11px] text-muted">
            <span className="truncate max-w-[110px]">{movie.genres[0] || 'Film'}</span>
            <span className="font-mono text-[10px]">{movie.release_year}</span>
          </div>
        </div>

        {/* Quick View Pill */}
        <div className="mt-2.5 pt-2 border-t border-border/50 flex items-center justify-between text-[10px] text-muted font-mono">
          <span className="flex items-center space-x-1 text-accent">
            <Play className="w-2.5 h-2.5 fill-accent" />
            <span>Details</span>
          </span>
          <span>{movie.rating_count ? `${movie.rating_count} ratings` : 'Curated'}</span>
        </div>
      </div>
    </div>
  );
};
