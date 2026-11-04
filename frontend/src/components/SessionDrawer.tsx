import React, { useState, useEffect } from 'react';
import { X, ThumbsUp, Bookmark, EyeOff, RefreshCw, Zap, ShieldCheck } from 'lucide-react';
import type { UserProfile } from '../types';
import { getUserHistory, getUserSaved } from '../api/client';

interface SessionDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  currentUser: UserProfile | null;
  sessionEventCount: number;
  onFeedback: (
    movieId: number,
    action: 'like' | 'dislike' | 'save' | 'dismiss' | 'rate',
    rating?: number
  ) => void;
  onRefreshFeed: () => void;
}

interface HistoryItem {
  movie_idx: number;
  movieId: number;
  title: string;
  release_year: number;
  genres: string[];
  rating_mean: number;
}

export const SessionDrawer: React.FC<SessionDrawerProps> = ({
  isOpen,
  onClose,
  currentUser,
  sessionEventCount,
  onFeedback,
  onRefreshFeed,
}) => {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [saved, setSaved] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'recent' | 'saved'>('recent');

  useEffect(() => {
    if (isOpen && currentUser) {
      setLoading(true);
      Promise.all([
        getUserHistory(currentUser.user_idx),
        getUserSaved(currentUser.user_idx),
      ])
        .then(([histData, savedData]) => {
          setHistory(histData || []);
          setSaved(savedData || []);
        })
        .catch(console.error)
        .finally(() => setLoading(false));
    }
  }, [isOpen, currentUser, sessionEventCount]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/70 backdrop-blur-sm animate-fade-in">
      <div className="w-full max-w-md bg-surface-elevated border-l border-border h-full flex flex-col shadow-2xl">
        {/* Header */}
        <div className="p-5 border-b border-border flex items-center justify-between bg-surface/80">
          <div className="flex items-center space-x-2.5">
            <div className="w-8 h-8 rounded-lg bg-accent/20 border border-accent/40 flex items-center justify-center text-accent">
              <Zap className="w-4 h-4" />
            </div>
            <div>
              <h3 className="font-display font-bold text-base text-white">
                Live Session Adaptation
              </h3>
              <p className="text-[11px] text-muted font-mono">
                {currentUser?.display_name} • {sessionEventCount} Session Events
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg bg-surface hover:bg-surface-hover text-muted hover:text-white border border-border transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Real-time Status Card */}
        <div className="p-4 mx-4 mt-4 rounded-xl bg-surface border border-accent/20 shadow-inner">
          <div className="flex items-center space-x-2 text-xs font-semibold text-accent mb-2">
            <ShieldCheck className="w-4 h-4" />
            <span>Fast Adaptation Active (Sub-1ms Latency)</span>
          </div>
          <p className="text-xs text-gray-300 leading-relaxed">
            Your clicks and ratings dynamically adjust your candidate retrieval vector. Disliked or dismissed titles are suppressed instantly from candidate pools without full model retraining.
          </p>

          <div className="grid grid-cols-2 gap-2 mt-3 pt-3 border-t border-border/60 text-center">
            <div className="p-2 rounded-lg bg-surface-elevated border border-border">
              <div className="text-base font-mono font-bold text-white">{sessionEventCount}</div>
              <div className="text-[10px] text-muted">Session Updates</div>
            </div>
            <div className="p-2 rounded-lg bg-surface-elevated border border-border">
              <div className="text-base font-mono font-bold text-accent">{saved.length}</div>
              <div className="text-[10px] text-muted">Watchlist Items</div>
            </div>
          </div>
        </div>

        {/* Tab Switcher */}
        <div className="flex px-4 mt-4 border-b border-border">
          <button
            onClick={() => setActiveTab('recent')}
            className={`flex-1 py-2 text-xs font-semibold border-b-2 transition ${
              activeTab === 'recent'
                ? 'border-accent text-accent'
                : 'border-transparent text-muted hover:text-white'
            }`}
          >
            Recent Clicks ({history.length})
          </button>
          <button
            onClick={() => setActiveTab('saved')}
            className={`flex-1 py-2 text-xs font-semibold border-b-2 transition ${
              activeTab === 'saved'
                ? 'border-accent text-accent'
                : 'border-transparent text-muted hover:text-white'
            }`}
          >
            Saved Watchlist ({saved.length})
          </button>
        </div>

        {/* Items List */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2.5">
          {loading ? (
            <div className="py-12 text-center text-muted">
              <RefreshCw className="w-6 h-6 animate-spin mx-auto mb-2 text-accent" />
              <p className="text-xs">Reading live session store...</p>
            </div>
          ) : activeTab === 'recent' ? (
            history.length > 0 ? (
              history.map((item, idx) => (
                <div
                  key={`hist-${item.movie_idx}-${idx}`}
                  className="p-3 rounded-xl bg-surface border border-border flex items-center justify-between hover:border-border-focus transition"
                >
                  <div className="min-w-0 pr-2">
                    <h5 className="text-xs font-semibold text-white truncate">
                      {item.title}
                    </h5>
                    <div className="flex items-center space-x-2 text-[10px] text-muted mt-0.5">
                      <span>{item.release_year}</span>
                      <span>•</span>
                      <span>{item.genres.slice(0, 2).join(', ')}</span>
                      <span>•</span>
                      <span className="text-amber-400 font-semibold">{item.rating_mean.toFixed(1)}★</span>
                    </div>
                  </div>

                  <div className="flex items-center space-x-1 flex-shrink-0">
                    <button
                      onClick={() => onFeedback(item.movie_idx, 'like')}
                      title="Boost Similar"
                      className="p-1.5 rounded-lg bg-surface-elevated hover:bg-surface-hover text-muted hover:text-amber-400 border border-border transition"
                    >
                      <ThumbsUp className="w-3.5 h-3.5" />
                    </button>
                    <button
                      onClick={() => onFeedback(item.movie_idx, 'dismiss')}
                      title="Suppress Title"
                      className="p-1.5 rounded-lg bg-surface-elevated hover:bg-red-500/20 text-muted hover:text-red-400 border border-border transition"
                    >
                      <EyeOff className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              ))
            ) : (
              <div className="py-12 text-center text-muted border border-dashed border-border rounded-xl">
                <ThumbsUp className="w-6 h-6 mx-auto mb-2 text-muted" />
                <p className="text-xs">No interactions yet in this session.</p>
                <p className="text-[10px] text-muted-dark mt-1">Like or rate movies on the home feed to adapt.</p>
              </div>
            )
          ) : (
            saved.length > 0 ? (
              saved.map((item) => (
                <div
                  key={`saved-${item.movie_idx}`}
                  className="p-3 rounded-xl bg-surface border border-border flex items-center justify-between hover:border-border-focus transition"
                >
                  <div className="min-w-0 pr-2">
                    <h5 className="text-xs font-semibold text-white truncate">
                      {item.title}
                    </h5>
                    <div className="flex items-center space-x-2 text-[10px] text-muted mt-0.5">
                      <span>{item.release_year}</span>
                      <span>•</span>
                      <span>{item.genres.slice(0, 2).join(', ')}</span>
                    </div>
                  </div>

                  <button
                    onClick={() => onFeedback(item.movie_idx, 'save')}
                    title="Remove from Watchlist"
                    className="p-1.5 rounded-lg bg-accent/20 text-accent hover:bg-red-500/20 hover:text-red-400 border border-accent/30 transition"
                  >
                    <Bookmark className="w-3.5 h-3.5 fill-accent" />
                  </button>
                </div>
              ))
            ) : (
              <div className="py-12 text-center text-muted border border-dashed border-border rounded-xl">
                <Bookmark className="w-6 h-6 mx-auto mb-2 text-muted" />
                <p className="text-xs">Watchlist is currently empty.</p>
                <p className="text-[10px] text-muted-dark mt-1">Click the bookmark icon on any movie to save it.</p>
              </div>
            )
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-border bg-surface/50">
          <button
            onClick={() => {
              onRefreshFeed();
              onClose();
            }}
            className="w-full py-2.5 rounded-xl bg-accent text-black font-semibold text-xs hover:bg-accent-hover transition flex items-center justify-center space-x-2 shadow-lg shadow-accent/10"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            <span>Apply Updates & Refresh Recommendations</span>
          </button>
        </div>
      </div>
    </div>
  );
};
