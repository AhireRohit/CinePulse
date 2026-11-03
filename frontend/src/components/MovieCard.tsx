import React, { useState } from 'react';
import { Star, ThumbsUp, ThumbsDown, Bookmark, EyeOff, Info, Sparkles } from 'lucide-react';
import type { RecommendationItem } from '../types';

interface MovieCardProps {
  movie: RecommendationItem;
  onOpenDetail: (movieId: number) => void;
  onFeedback: (movieId: number, action: 'like' | 'dislike' | 'save' | 'dismiss') => void;
  isSaved?: boolean;
  isLiked?: boolean;
}

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

  if (dismissed) {
    return null;
  }

  // Generate deterministic subtle gradient palette based on movie ID for typographical poster
  const palettes = [
    'from-slate-900 via-neutral-900 to-stone-950',
    'from-zinc-900 via-stone-900 to-neutral-950',
    'from-neutral-900 via-slate-900 to-gray-950',
    'from-stone-900 via-zinc-900 to-slate-950',
  ];
  const palette = palettes[movie.movie_idx % palettes.length];

  const handleAction = (
    e: React.MouseEvent,
    action: 'like' | 'dislike' | 'save' | 'dismiss'
  ) => {
    e.stopPropagation();
    if (action === 'like') {
      setLiked(!liked);
    } else if (action === 'save') {
      setSaved(!saved);
    } else if (action === 'dismiss' || action === 'dislike') {
      setDismissed(true);
    }
    onFeedback(movie.movie_idx, action);
  };

  return (
    <div
      onClick={() => onOpenDetail(movie.movie_idx)}
      className="group relative flex-shrink-0 w-48 sm:w-56 cursor-pointer select-none rounded-xl bg-surface border border-border hover:border-border-focus hover:shadow-xl hover:shadow-black/50 transition-all duration-300 transform hover:-translate-y-1 overflow-hidden"
    >
      {/* Poster / Typographic Art Cover */}
      <div className={`relative h-64 sm:h-72 w-full bg-gradient-to-br ${palette} p-4 flex flex-col justify-between border-b border-border/50`}>
        {/* Top Badges */}
        <div className="flex items-center justify-between z-10">
          <div className="flex items-center space-x-1 px-2 py-0.5 rounded-full bg-black/60 backdrop-blur-sm border border-white/10 text-[11px] font-semibold text-white">
            <Star className="w-3 h-3 text-amber-400 fill-amber-400" />
            <span>{movie.rating_mean.toFixed(1)}</span>
          </div>

          {movie.novelty_score > 0.55 && (
            <div className="flex items-center space-x-1 px-2 py-0.5 rounded-full bg-accent/20 border border-accent/40 text-[10px] font-medium text-accent">
              <Sparkles className="w-2.5 h-2.5" />
              <span>Deep Cut</span>
            </div>
          )}
        </div>

        {/* Center Title Display */}
        <div className="my-auto z-10 text-center px-2">
          <h4 className="font-display font-bold text-lg text-gray-100 tracking-tight leading-snug line-clamp-3 group-hover:text-accent transition-colors">
            {movie.title}
          </h4>
          <span className="block mt-1 font-mono text-xs text-muted">
            {movie.release_year}
          </span>
        </div>

        {/* Bottom Recommendation Reason Pill */}
        <div className="z-10">
          <div className="px-2 py-1 rounded bg-black/70 backdrop-blur-sm border border-white/10 text-[10px] text-gray-300 font-medium truncate">
            {movie.reason}
          </div>
        </div>

        {/* Action Overlay on Hover */}
        <div className="absolute inset-0 bg-black/80 backdrop-blur-sm opacity-0 group-hover:opacity-100 transition-opacity duration-200 flex flex-col justify-between p-4 z-20">
          <div className="text-right">
            <button
              onClick={(e) => handleAction(e, 'dismiss')}
              title="Dismiss / Don't recommend again"
              className="p-1.5 rounded-full bg-surface-elevated hover:bg-red-500/20 text-muted hover:text-red-400 transition"
            >
              <EyeOff className="w-4 h-4" />
            </button>
          </div>

          <div className="text-center px-1">
            <p className="text-xs text-gray-300 line-clamp-3 mb-2">
              {movie.full_title}
            </p>
            <div className="flex flex-wrap justify-center gap-1 mb-2">
              {movie.genres.slice(0, 3).map((g) => (
                <span
                  key={g}
                  className="px-1.5 py-0.5 rounded text-[10px] bg-surface-elevated text-gray-300 border border-border"
                >
                  {g}
                </span>
              ))}
            </div>
          </div>

          <div className="flex items-center justify-around pt-2 border-t border-border">
            <button
              onClick={(e) => handleAction(e, 'like')}
              title="Like"
              className={`p-2 rounded-lg transition ${
                liked
                  ? 'bg-amber-500/20 text-amber-400'
                  : 'bg-surface-elevated hover:bg-surface-hover text-gray-300'
              }`}
            >
              <ThumbsUp className={`w-4 h-4 ${liked ? 'fill-amber-400' : ''}`} />
            </button>

            <button
              onClick={(e) => handleAction(e, 'dislike')}
              title="Dislike"
              className="p-2 rounded-lg bg-surface-elevated hover:bg-surface-hover text-gray-300 hover:text-red-400 transition"
            >
              <ThumbsDown className="w-4 h-4" />
            </button>

            <button
              onClick={(e) => handleAction(e, 'save')}
              title="Save to Library"
              className={`p-2 rounded-lg transition ${
                saved
                  ? 'bg-accent/20 text-accent'
                  : 'bg-surface-elevated hover:bg-surface-hover text-gray-300'
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
              className="p-2 rounded-lg bg-surface-elevated hover:bg-surface-hover text-gray-300 hover:text-white transition"
            >
              <Info className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      {/* Card Info Below */}
      <div className="p-3 bg-surface">
        <h5 className="font-medium text-xs text-gray-200 truncate group-hover:text-white transition">
          {movie.title}
        </h5>
        <div className="mt-1 flex items-center justify-between text-[11px] text-muted">
          <span>{movie.genres[0] || 'Film'}</span>
          <span>{movie.release_year}</span>
        </div>
      </div>
    </div>
  );
};
