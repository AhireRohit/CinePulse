import React, { useEffect, useState } from 'react';
import { X, Star, ThumbsUp, ThumbsDown, Bookmark, EyeOff, Sparkles } from 'lucide-react';
import type { MovieDetail } from '../types';
import { getMovie } from '../api/client';

interface MovieDetailModalProps {
  movieId: number | null;
  onClose: () => void;
  onFeedback: (movieId: number, action: 'like' | 'dislike' | 'save' | 'dismiss' | 'rate', rating?: number) => void;
}

export const MovieDetailModal: React.FC<MovieDetailModalProps> = ({
  movieId,
  onClose,
  onFeedback,
}) => {
  const [detail, setDetail] = useState<MovieDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [userRating, setUserRating] = useState<number | null>(null);

  useEffect(() => {
    if (movieId === null) return;
    setLoading(true);
    getMovie(movieId)
      .then((data) => setDetail(data))
      .catch((err) => console.error(err))
      .finally(() => setLoading(false));
  }, [movieId]);

  if (movieId === null) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md overflow-y-auto">
      <div className="relative w-full max-w-3xl bg-surface-elevated border border-border rounded-2xl shadow-2xl overflow-hidden my-8">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 z-20 p-2 rounded-full bg-surface/80 hover:bg-surface text-muted hover:text-white border border-border transition"
        >
          <X className="w-5 h-5" />
        </button>

        {loading || !detail ? (
          <div className="p-16 text-center text-muted">
            <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-sm">Fetching model diagnostics and movie features...</p>
          </div>
        ) : (
          <div>
            {/* Header Art / Typographic Banner */}
            <div className="relative bg-gradient-to-r from-neutral-900 via-stone-900 to-slate-900 p-8 border-b border-border">
              <div className="flex flex-wrap items-center gap-2 mb-3">
                {detail.genres.map((g) => (
                  <span
                    key={g}
                    className="px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-black/50 border border-white/10 text-gray-200"
                  >
                    {g}
                  </span>
                ))}
                <span className="text-xs font-mono text-muted">• {detail.release_year}</span>
              </div>

              <h2 className="font-display font-bold text-3xl text-white tracking-tight leading-snug">
                {detail.title}
              </h2>
              <p className="text-xs text-muted font-mono mt-1">{detail.full_title}</p>

              {/* Ratings and Stats */}
              <div className="flex flex-wrap items-center gap-6 mt-6 pt-4 border-t border-white/10">
                <div className="flex items-center space-x-2">
                  <Star className="w-5 h-5 text-amber-400 fill-amber-400" />
                  <div>
                    <div className="text-sm font-bold text-white">
                      {detail.rating_mean.toFixed(2)}{' '}
                      <span className="text-xs font-normal text-muted">/ 5.0</span>
                    </div>
                    <div className="text-[10px] text-muted font-mono">
                      {detail.rating_count.toLocaleString()} Ratings
                    </div>
                  </div>
                </div>

                <div className="h-8 w-px bg-white/10 hidden sm:block" />

                <div>
                  <div className="text-xs font-mono text-gray-200">
                    {detail.bayesian_rating.toFixed(2)}
                  </div>
                  <div className="text-[10px] text-muted font-mono">Bayesian Weighted Score</div>
                </div>

                <div className="h-8 w-px bg-white/10 hidden sm:block" />

                <div>
                  <div className="text-xs font-mono text-accent">MovieLens 1M</div>
                  <div className="text-[10px] text-muted font-mono">Dataset Item ID #{detail.movieId}</div>
                </div>
              </div>
            </div>

            {/* Body Content */}
            <div className="p-8 space-y-6">
              {/* Overview */}
              <div>
                <h4 className="text-xs font-mono uppercase tracking-wider text-muted mb-2">
                  Catalog Synopsis
                </h4>
                <p className="text-sm text-gray-300 leading-relaxed">
                  {detail.overview}
                </p>
              </div>

              {/* User Rating & Feedback Interaction */}
              <div className="p-4 rounded-xl bg-surface border border-border">
                <div className="flex flex-wrap items-center justify-between gap-4">
                  <div>
                    <h5 className="text-xs font-medium text-white mb-1">
                      Rate or React to Train Session
                    </h5>
                    <p className="text-[11px] text-muted">
                      Your rating instantly updates session vectors and candidate reranking.
                    </p>
                  </div>

                  {/* 5-Star Rating Buttons */}
                  <div className="flex items-center space-x-1">
                    {[1, 2, 3, 4, 5].map((star) => (
                      <button
                        key={star}
                        onClick={() => {
                          setUserRating(star);
                          onFeedback(detail.movie_idx, 'rate', star);
                        }}
                        className="p-1 hover:scale-110 transition"
                      >
                        <Star
                          className={`w-5 h-5 ${
                            (userRating || 0) >= star
                              ? 'text-amber-400 fill-amber-400'
                              : 'text-muted-dark hover:text-amber-400'
                          }`}
                        />
                      </button>
                    ))}
                  </div>

                  {/* Quick Action Reactions */}
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => onFeedback(detail.movie_idx, 'like')}
                      className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-surface-elevated hover:bg-surface-hover border border-border text-xs text-gray-200 transition"
                    >
                      <ThumbsUp className="w-3.5 h-3.5" />
                      <span>Like</span>
                    </button>

                    <button
                      onClick={() => onFeedback(detail.movie_idx, 'dislike')}
                      className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-surface-elevated hover:bg-surface-hover border border-border text-xs text-gray-200 hover:text-red-400 transition"
                    >
                      <ThumbsDown className="w-3.5 h-3.5" />
                      <span>Dislike</span>
                    </button>

                    <button
                      onClick={() => onFeedback(detail.movie_idx, 'save')}
                      className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-surface-elevated hover:bg-surface-hover border border-border text-xs text-accent transition"
                    >
                      <Bookmark className="w-3.5 h-3.5" />
                      <span>Save</span>
                    </button>

                    <button
                      onClick={() => {
                        onFeedback(detail.movie_idx, 'dismiss');
                        onClose();
                      }}
                      className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-surface-elevated hover:bg-red-500/10 border border-border text-xs text-muted hover:text-red-400 transition"
                    >
                      <EyeOff className="w-3.5 h-3.5" />
                      <span>Dismiss</span>
                    </button>
                  </div>
                </div>
              </div>

              {/* Similar Movies Rail */}
              {detail.similar_movies && detail.similar_movies.length > 0 && (
                <div>
                  <h4 className="text-xs font-mono uppercase tracking-wider text-muted mb-3 flex items-center space-x-1.5">
                    <Sparkles className="w-3.5 h-3.5 text-accent" />
                    <span>Collaborative Filtering Neighbors (Item-CF Similarity)</span>
                  </h4>
                  <div className="flex space-x-4 overflow-x-auto pb-2 scrollbar-none">
                    {detail.similar_movies.map((sim) => (
                      <div
                        key={sim.movie_idx}
                        onClick={() => {
                          setDetail(null);
                          getMovie(sim.movie_idx).then(setDetail);
                        }}
                        className="flex-shrink-0 w-36 p-3 rounded-lg bg-surface border border-border hover:border-border-focus cursor-pointer transition"
                      >
                        <h6 className="font-medium text-xs text-white truncate">{sim.title}</h6>
                        <div className="flex items-center justify-between text-[10px] text-muted mt-1 font-mono">
                          <span>{sim.release_year}</span>
                          <span className="text-accent font-semibold">{sim.final_score.toFixed(2)} Sim</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
