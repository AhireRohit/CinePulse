import React, { useState } from 'react';
import { X, Check, Film, Sparkles } from 'lucide-react';
import { createUser } from '../api/client';
import type { UserProfile } from '../types';

interface OnboardingModalProps {
  onClose: () => void;
  onUserCreated: (newUser: UserProfile) => void;
}

const AVAILABLE_GENRES = [
  'Sci-Fi', 'Mystery', 'Crime', 'Film-Noir', 'Drama', 'Thriller',
  'Action', 'Adventure', 'Animation', 'Comedy', 'Fantasy', 'Romance',
  'Horror', 'Documentary', 'War', 'Western'
];

const SEED_MOVIES = [
  { movie_idx: 257, title: 'Star Wars: A New Hope', year: 1977, genre: 'Sci-Fi' },
  { movie_idx: 110, title: 'Pulp Fiction', year: 1994, genre: 'Crime' },
  { movie_idx: 293, title: 'The Matrix', year: 1999, genre: 'Sci-Fi' },
  { movie_idx: 315, title: 'The Shawshank Redemption', year: 1994, genre: 'Drama' },
  { movie_idx: 50, title: 'The Usual Suspects', year: 1995, genre: 'Mystery' },
  { movie_idx: 535, title: 'Blade Runner', year: 1982, genre: 'Sci-Fi' },
  { movie_idx: 846, title: 'The Godfather', year: 1972, genre: 'Crime' },
  { movie_idx: 911, title: 'Casablanca', year: 1942, genre: 'Drama' },
  { movie_idx: 924, title: '2001: A Space Odyssey', year: 1968, genre: 'Sci-Fi' },
  { movie_idx: 1196, title: 'Alien', year: 1979, genre: 'Horror' },
];

const AVATAR_COLORS = ['#3B82F6', '#D4AF37', '#10B981', '#EC4899', '#8B5CF6', '#EF4444'];

export const OnboardingModal: React.FC<OnboardingModalProps> = ({
  onClose,
  onUserCreated,
}) => {
  const [displayName, setDisplayName] = useState('');
  const [username, setUsername] = useState('');
  const [avatarColor, setAvatarColor] = useState(AVATAR_COLORS[0]);
  const [selectedGenres, setSelectedGenres] = useState<string[]>([]);
  const [selectedMovies, setSelectedMovies] = useState<number[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const toggleGenre = (g: string) => {
    if (selectedGenres.includes(g)) {
      setSelectedGenres(selectedGenres.filter((item) => item !== g));
    } else {
      setSelectedGenres([...selectedGenres, g]);
    }
  };

  const toggleMovie = (idx: number) => {
    if (selectedMovies.includes(idx)) {
      setSelectedMovies(selectedMovies.filter((item) => item !== idx));
    } else {
      setSelectedMovies([...selectedMovies, idx]);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!displayName.trim() || !username.trim()) {
      setError('Please provide a name and username.');
      return;
    }
    if (selectedGenres.length === 0) {
      setError('Select at least one genre to calibrate your cold-start profile.');
      return;
    }

    setSubmitting(true);
    setError(null);

    try {
      const cleanUsername = username.toLowerCase().replace(/[^a-z0-9_]/g, '');
      const user = await createUser({
        username: cleanUsername,
        display_name: displayName,
        preferred_genres: selectedGenres,
        favorite_movies: selectedMovies,
        avatar_color: avatarColor,
      });
      onUserCreated(user);
      onClose();
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to create profile.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md overflow-y-auto">
      <div className="relative w-full max-w-2xl bg-surface-elevated border border-border rounded-2xl shadow-2xl p-6 sm:p-8 my-8">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-2 rounded-full bg-surface hover:bg-surface-hover text-muted hover:text-white transition"
        >
          <X className="w-5 h-5" />
        </button>

        <div className="mb-6">
          <div className="flex items-center space-x-2 text-xs font-mono text-accent uppercase tracking-wider mb-1">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Cold-Start Preference Calibration</span>
          </div>
          <h2 className="font-display font-bold text-2xl text-white">
            Create Custom Profile
          </h2>
          <p className="text-xs text-muted mt-1">
            Build your personal recommendation vector. We use zero-shot genre projection and seed film affinity to bootstrap recommendations without requiring historical dataset logs.
          </p>
        </div>

        {error && (
          <div className="mb-4 p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-xs text-red-400">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-xs font-medium text-gray-300 mb-1.5">
                Display Name
              </label>
              <input
                type="text"
                placeholder="e.g. Maya Lin"
                value={displayName}
                onChange={(e) => {
                  setDisplayName(e.target.value);
                  if (!username) {
                    setUsername(e.target.value.toLowerCase().replace(/\s+/g, '_'));
                  }
                }}
                className="w-full px-3.5 py-2 rounded-lg bg-surface border border-border focus:border-accent text-white text-xs outline-none transition"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-gray-300 mb-1.5">
                Username / Handle
              </label>
              <input
                type="text"
                placeholder="e.g. maya_lin"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full px-3.5 py-2 rounded-lg bg-surface border border-border focus:border-accent text-white text-xs outline-none transition"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-medium text-gray-300 mb-2">
              Profile Palette
            </label>
            <div className="flex items-center space-x-3">
              {AVATAR_COLORS.map((c) => (
                <div
                  key={c}
                  onClick={() => setAvatarColor(c)}
                  className={`w-7 h-7 rounded-full cursor-pointer flex items-center justify-center transition ${
                    avatarColor === c ? 'ring-2 ring-white scale-110' : 'opacity-70 hover:opacity-100'
                  }`}
                  style={{ backgroundColor: c }}
                >
                  {avatarColor === c && <Check className="w-3.5 h-3.5 text-white" />}
                </div>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-xs font-medium text-gray-300 mb-2">
              Select Preferred Genres <span className="text-muted">({selectedGenres.length} selected)</span>
            </label>
            <div className="flex flex-wrap gap-2">
              {AVAILABLE_GENRES.map((g) => {
                const isSelected = selectedGenres.includes(g);
                return (
                  <button
                    key={g}
                    type="button"
                    onClick={() => toggleGenre(g)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-medium transition ${
                      isSelected
                        ? 'bg-accent text-black font-semibold shadow-md shadow-accent/10'
                        : 'bg-surface hover:bg-surface-hover text-gray-300 border border-border'
                    }`}
                  >
                    {g}
                  </button>
                );
              })}
            </div>
          </div>

          <div>
            <label className="block text-xs font-medium text-gray-300 mb-2">
              Seed Movies You Love <span className="text-muted">(Optional bootstrap)</span>
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-h-48 overflow-y-auto pr-1">
              {SEED_MOVIES.map((m) => {
                const isSelected = selectedMovies.includes(m.movie_idx);
                return (
                  <div
                    key={m.movie_idx}
                    onClick={() => toggleMovie(m.movie_idx)}
                    className={`flex items-center justify-between p-2.5 rounded-lg border cursor-pointer transition ${
                      isSelected
                        ? 'bg-amber-500/10 border-amber-500/40 text-white'
                        : 'bg-surface hover:bg-surface-hover border-border text-gray-300'
                    }`}
                  >
                    <div className="flex items-center space-x-2 truncate">
                      <Film className={`w-3.5 h-3.5 flex-shrink-0 ${isSelected ? 'text-accent' : 'text-muted'}`} />
                      <div className="truncate">
                        <div className="text-xs font-medium truncate">{m.title}</div>
                        <div className="text-[10px] text-muted">{m.genre} • {m.year}</div>
                      </div>
                    </div>
                    {isSelected && <Check className="w-4 h-4 text-accent flex-shrink-0 ml-2" />}
                  </div>
                );
              })}
            </div>
          </div>

          <div className="pt-4 border-t border-border flex items-center justify-end space-x-3">
            <button
              type="button"
              onClick={onClose}
              className="px-4 py-2 rounded-lg text-xs text-muted hover:text-white transition"
            >
              Cancel
            </button>

            <button
              type="submit"
              disabled={submitting}
              className="px-6 py-2 rounded-lg bg-accent hover:bg-accent-hover text-black font-semibold text-xs transition shadow-lg shadow-accent/15 disabled:opacity-50"
            >
              {submitting ? 'Calibrating Vector...' : 'Initialize & Launch Profile'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
