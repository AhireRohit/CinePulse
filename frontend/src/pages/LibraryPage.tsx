import React, { useState, useEffect } from 'react';
import { Bookmark, Clock, Film, Trash2, Star, User } from 'lucide-react';
import type { UserProfile } from '../types';
import { getUserHistory, getUserSaved } from '../api/client';

interface LibraryPageProps {
  currentUser: UserProfile | null;
  onOpenDetail: (movieId: number) => void;
  onFeedback: (movieId: number, action: 'like' | 'dislike' | 'save' | 'dismiss') => void;
}

export const LibraryPage: React.FC<LibraryPageProps> = ({
  currentUser,
  onOpenDetail,
  onFeedback,
}) => {
  const [activeTab, setActiveTab] = useState<'saved' | 'history'>('saved');
  const [savedMovies, setSavedMovies] = useState<any[]>([]);
  const [historyMovies, setHistoryMovies] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!currentUser) return;
    setLoading(true);
    Promise.all([
      getUserSaved(currentUser.user_idx),
      getUserHistory(currentUser.user_idx),
    ])
      .then(([saved, history]) => {
        setSavedMovies(saved);
        setHistoryMovies(history);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [currentUser]);

  if (!currentUser) {
    return (
      <div className="max-w-7xl mx-auto px-6 py-16 text-center text-muted">
        <User className="w-10 h-10 mx-auto mb-2 text-muted" />
        <p>Please select or create a profile to view library.</p>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Profile Overview Card */}
      <div className="p-6 sm:p-8 rounded-2xl bg-surface border border-border mb-8 flex flex-wrap items-center justify-between gap-6">
        <div className="flex items-center space-x-4">
          <div
            className="w-16 h-16 rounded-2xl flex items-center justify-center text-2xl font-bold text-white shadow-xl"
            style={{ backgroundColor: currentUser.avatar_color }}
          >
            {currentUser.display_name.charAt(0)}
          </div>
          <div>
            <h2 className="font-display font-bold text-2xl text-white">
              {currentUser.display_name}
            </h2>
            <p className="text-xs text-muted font-mono mt-0.5">
              @{currentUser.username} • User Index #{currentUser.user_idx}
            </p>
            {currentUser.bio && (
              <p className="text-xs text-gray-300 mt-2 max-w-xl">{currentUser.bio}</p>
            )}
          </div>
        </div>

        {/* Stats Pills */}
        <div className="flex items-center space-x-4">
          <div className="px-4 py-2.5 rounded-xl bg-surface-elevated border border-border text-center">
            <div className="font-display font-bold text-lg text-white">
              {savedMovies.length}
            </div>
            <div className="text-[10px] uppercase font-mono text-muted tracking-wider">
              Saved Movies
            </div>
          </div>

          <div className="px-4 py-2.5 rounded-xl bg-surface-elevated border border-border text-center">
            <div className="font-display font-bold text-lg text-white">
              {historyMovies.length}
            </div>
            <div className="text-[10px] uppercase font-mono text-muted tracking-wider">
              Watched & Rated
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex items-center space-x-4 border-b border-border mb-6">
        <button
          onClick={() => setActiveTab('saved')}
          className={`flex items-center space-x-2 py-3 text-xs font-semibold border-b-2 transition ${
            activeTab === 'saved'
              ? 'border-accent text-accent'
              : 'border-transparent text-muted hover:text-white'
          }`}
        >
          <Bookmark className="w-4 h-4" />
          <span>Saved to Watchlist ({savedMovies.length})</span>
        </button>

        <button
          onClick={() => setActiveTab('history')}
          className={`flex items-center space-x-2 py-3 text-xs font-semibold border-b-2 transition ${
            activeTab === 'history'
              ? 'border-accent text-accent'
              : 'border-transparent text-muted hover:text-white'
          }`}
        >
          <Clock className="w-4 h-4" />
          <span>Interaction History ({historyMovies.length})</span>
        </button>
      </div>

      {/* Tab Content */}
      {loading ? (
        <div className="py-20 text-center text-muted">
          <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <p className="text-xs">Loading user library items...</p>
        </div>
      ) : activeTab === 'saved' ? (
        savedMovies.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
            {savedMovies.map((m) => (
              <div
                key={`saved-${m.movie_idx}`}
                onClick={() => onOpenDetail(m.movie_idx)}
                className="p-4 rounded-xl bg-surface border border-border hover:border-border-focus flex items-center justify-between cursor-pointer group transition"
              >
                <div className="flex items-center space-x-3 truncate">
                  <div className="w-10 h-10 rounded-lg bg-surface-elevated border border-border flex items-center justify-center text-accent flex-shrink-0">
                    <Film className="w-5 h-5" />
                  </div>
                  <div className="truncate">
                    <h5 className="font-medium text-xs text-white truncate group-hover:text-accent transition">
                      {m.title}
                    </h5>
                    <div className="text-[10px] text-muted font-mono mt-0.5">
                      {m.genres?.slice(0, 2).join(', ')} • {m.release_year}
                    </div>
                  </div>
                </div>

                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onFeedback(m.movie_idx, 'dismiss');
                    setSavedMovies(savedMovies.filter((item) => item.movie_idx !== m.movie_idx));
                  }}
                  title="Remove from saved"
                  className="p-2 rounded-lg text-muted hover:text-red-400 hover:bg-surface-elevated transition flex-shrink-0"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        ) : (
          <div className="py-16 text-center text-muted border border-dashed border-border rounded-xl">
            <Bookmark className="w-8 h-8 text-muted mx-auto mb-2" />
            <p className="text-sm text-gray-300">No saved movies yet.</p>
            <p className="text-xs text-muted mt-1">
              Click the bookmark icon on any movie card to add it to your personal watchlist.
            </p>
          </div>
        )
      ) : historyMovies.length > 0 ? (
        <div className="space-y-3">
          {historyMovies.map((m, idx) => (
            <div
              key={`hist-${m.movie_idx}-${idx}`}
              onClick={() => onOpenDetail(m.movie_idx)}
              className="p-3.5 rounded-xl bg-surface border border-border hover:border-border-focus flex items-center justify-between cursor-pointer group transition"
            >
              <div className="flex items-center space-x-3 truncate">
                <span className="font-mono text-xs text-muted w-6 text-right">
                  #{idx + 1}
                </span>
                <div className="truncate">
                  <h5 className="font-medium text-xs text-white truncate group-hover:text-accent transition">
                    {m.title}
                  </h5>
                  <div className="text-[10px] text-muted font-mono mt-0.5">
                    {m.genres?.join(', ')} • {m.release_year}
                  </div>
                </div>
              </div>

              <div className="flex items-center space-x-1 text-xs text-amber-400 font-medium">
                <Star className="w-3.5 h-3.5 fill-amber-400" />
                <span>{m.rating_mean?.toFixed(1) || '4.0'}</span>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="py-16 text-center text-muted border border-dashed border-border rounded-xl">
          <Clock className="w-8 h-8 text-muted mx-auto mb-2" />
          <p className="text-sm text-gray-300">No interaction history recorded yet.</p>
        </div>
      )}
    </div>
  );
};
