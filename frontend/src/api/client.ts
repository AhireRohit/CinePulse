import axios from 'axios';
import type {
  UserProfile,
  RecommendationFeed,
  MovieDetail,
  UserInsights,
  ModelEvaluation,
  RecommendationItem,
} from '../types';

const API_BASE = import.meta.env.VITE_API_BASE || '';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 3000,
  headers: {
    'Content-Type': 'application/json',
  },
});

const getBaseUrl = () => {
  const base = import.meta.env.BASE_URL || './';
  return base.endsWith('/') ? base : `${base}/`;
};

const isStaticHost =
  typeof window !== 'undefined' &&
  (window.location.hostname.endsWith('github.io') ||
    window.location.protocol === 'file:' ||
    window.location.hostname === 'localhost' && Boolean(import.meta.env.VITE_DEMO_MODE));

// Cache movies in memory for fast search and details in static mode
let cachedMovies: any[] | null = null;
const loadMovies = async (): Promise<any[]> => {
  if (cachedMovies && cachedMovies.length > 0) return cachedMovies;
  try {
    const res = await fetch(`${getBaseUrl()}demo-data/movies.json`);
    cachedMovies = await res.json();
    return cachedMovies || [];
  } catch (err) {
    console.warn('Could not load demo movies catalog:', err);
    return [];
  }
};

export const getHealth = async () => {
  if (!isStaticHost) {
    try {
      const res = await api.get('/health');
      return res.data;
    } catch {}
  }
  return { status: 'healthy', mode: 'static_showcase', engine: 'precomputed_v2' };
};

export const getUsers = async (): Promise<UserProfile[]> => {
  if (!isStaticHost) {
    try {
      const res = await api.get('/api/users');
      if (res.data && Array.isArray(res.data) && res.data.length > 0) {
        return res.data;
      }
    } catch {}
  }
  try {
    const res = await fetch(`${getBaseUrl()}demo-data/users.json`);
    const demoUsers: UserProfile[] = await res.json();
    const customUsersStr = localStorage.getItem('cinepulse_custom_users');
    if (customUsersStr) {
      try {
        const customUsers = JSON.parse(customUsersStr);
        return [...demoUsers, ...customUsers];
      } catch {}
    }
    return demoUsers;
  } catch (err) {
    console.error('Failed to load users from demo data:', err);
    return [];
  }
};

export const createUser = async (data: {
  username: string;
  display_name: string;
  preferred_genres: string[];
  favorite_movies: number[];
  avatar_color?: string;
  bio?: string;
}): Promise<UserProfile> => {
  if (!isStaticHost) {
    try {
      const res = await api.post('/api/users', data);
      return res.data;
    } catch {}
  }

  const customUser: UserProfile = {
    user_idx: 9000 + Math.floor(Math.random() * 1000),
    username: data.username,
    display_name: data.display_name,
    avatar_color: data.avatar_color || '#3B82F6',
    bio: data.bio || `Passionate enthusiast of ${data.preferred_genres.join(', ')}.`,
    preferred_genres: data.preferred_genres,
  };
  const customUsersStr = localStorage.getItem('cinepulse_custom_users');
  const customUsers = customUsersStr ? JSON.parse(customUsersStr) : [];
  customUsers.push(customUser);
  localStorage.setItem('cinepulse_custom_users', JSON.stringify(customUsers));
  return customUser;
};

export const getRecommendations = async (
  userId: number,
  params?: { k?: number; diversity?: number; novelty?: number }
): Promise<RecommendationFeed> => {
  if (!isStaticHost) {
    try {
      const res = await api.get(`/api/recommendations/${userId}`, { params });
      if (res.data && res.data.rails) {
        return res.data;
      }
    } catch {}
  }

  const availableDemoIds = [12, 45, 88, 120];
  const targetId = availableDemoIds.includes(userId) ? userId : 12;
  const res = await fetch(`${getBaseUrl()}demo-data/recommendations_${targetId}.json`);
  const feed: RecommendationFeed = await res.json();

  // Apply real-time localStorage session adaptation
  const dismissedStr = localStorage.getItem(`cinepulse_dismissed_${userId}`);
  const dislikedStr = localStorage.getItem(`cinepulse_disliked_${userId}`);
  const hiddenIds = new Set<number>();
  if (dismissedStr) {
    try {
      JSON.parse(dismissedStr).forEach((id: number) => hiddenIds.add(id));
    } catch {}
  }
  if (dislikedStr) {
    try {
      JSON.parse(dislikedStr).forEach((id: number) => hiddenIds.add(id));
    } catch {}
  }

  const filteredRails: Record<string, RecommendationItem[]> = {};
  for (const [railName, items] of Object.entries(feed.rails)) {
    filteredRails[railName] = items.filter(
      (item) => !hiddenIds.has(item.movie_idx) && !hiddenIds.has(item.movieId)
    );
  }

  let filteredHero = feed.hero_featured;
  if (
    filteredHero &&
    (hiddenIds.has(filteredHero.movie_idx) || hiddenIds.has(filteredHero.movieId))
  ) {
    filteredHero = undefined;
  }

  return {
    ...feed,
    user_idx: userId,
    hero_featured: filteredHero,
    rails: filteredRails,
  };
};

export const postFeedback = async (
  userId: number,
  movieId: number,
  action: 'like' | 'dislike' | 'save' | 'dismiss' | 'rate',
  rating?: number
) => {
  if (!isStaticHost) {
    try {
      const res = await api.post('/api/feedback', {
        user_idx: userId,
        movie_idx: movieId,
        action,
        rating,
      });
      return res.data;
    } catch {}
  }

  const keyMap: Record<string, string> = {
    save: `cinepulse_saved_${userId}`,
    dislike: `cinepulse_disliked_${userId}`,
    dismiss: `cinepulse_dismissed_${userId}`,
    like: `cinepulse_liked_${userId}`,
    rate: `cinepulse_rated_${userId}`,
  };
  const key = keyMap[action] || `cinepulse_events_${userId}`;
  const storedStr = localStorage.getItem(key);
  const list: number[] = storedStr ? JSON.parse(storedStr) : [];
  if (!list.includes(movieId)) {
    list.push(movieId);
    localStorage.setItem(key, JSON.stringify(list));
  }

  return {
    success: true,
    user_idx: userId,
    movie_idx: movieId,
    action,
    message: `Recorded '${action}' in active session. Demo account baseline remains protected.`,
  };
};

export const getMovie = async (movieId: number): Promise<MovieDetail> => {
  if (!isStaticHost) {
    try {
      const res = await api.get(`/api/movies/${movieId}`);
      if (res.data) return res.data;
    } catch {}
  }

  const movies = await loadMovies();
  const movie =
    movies.find((m: any) => m.movie_idx === movieId || m.movieId === movieId) || movies[0];

  const genresSet = new Set(movie.genres || []);
  const similar = movies
    .filter(
      (m: any) =>
        m.movie_idx !== movie.movie_idx &&
        m.genres?.some((g: string) => genresSet.has(g))
    )
    .slice(0, 6)
    .map((m: any, idx: number) => ({
      rank: idx + 1,
      movie_idx: m.movie_idx,
      movieId: m.movieId,
      title: m.title,
      full_title: m.full_title,
      release_year: m.release_year,
      genres: m.genres,
      rating_mean: m.rating_mean,
      rating_count: m.rating_count,
      bayesian_rating: m.bayesian_rating,
      final_score: 0.85 - idx * 0.05,
      relevance_score: 0.85 - idx * 0.05,
      diversity_penalty: 0.0,
      novelty_score: 0.2,
      retrieval_source: 'item_cf',
      reason: 'Similar themes & genres',
      poster_url: m.poster_url,
    }));

  return {
    movie_idx: movie.movie_idx,
    movieId: movie.movieId,
    title: movie.title,
    full_title: movie.full_title,
    release_year: movie.release_year,
    genres: movie.genres,
    rating_mean: movie.rating_mean,
    rating_count: movie.rating_count,
    bayesian_rating: movie.bayesian_rating,
    poster_url: movie.poster_url,
    overview: `A critically noted ${movie.release_year} release featuring ${(movie.genres || []).join(', ')}. Rated ${movie.rating_mean}/5.0 across ${movie.rating_count.toLocaleString()} community reviews.`,
    similar_movies: similar,
  };
};

export const searchMovies = async (q: string): Promise<RecommendationItem[]> => {
  if (!isStaticHost) {
    try {
      const res = await api.get('/api/search', { params: { q } });
      if (res.data?.results) return res.data.results;
    } catch {}
  }

  const movies = await loadMovies();
  const queryLower = q.toLowerCase().trim();
  const matches = movies
    .filter((m: any) => {
      const titleMatch = m.title?.toLowerCase().includes(queryLower);
      const fullTitleMatch = m.full_title?.toLowerCase().includes(queryLower);
      const genreMatch = m.genres?.some((g: string) =>
        g.toLowerCase().includes(queryLower)
      );
      const yearMatch = String(m.release_year).includes(queryLower);
      return titleMatch || fullTitleMatch || genreMatch || yearMatch;
    })
    .slice(0, 30);

  return matches.map((m: any, idx: number) => ({
    rank: idx + 1,
    movie_idx: m.movie_idx,
    movieId: m.movieId,
    title: m.title,
    full_title: m.full_title,
    release_year: m.release_year,
    genres: m.genres,
    rating_mean: m.rating_mean,
    rating_count: m.rating_count,
    bayesian_rating: m.bayesian_rating,
    final_score: m.bayesian_rating,
    relevance_score: m.rating_mean,
    diversity_penalty: 0.0,
    novelty_score: 0.3,
    retrieval_source: 'search',
    reason: 'Search query match',
    poster_url: m.poster_url,
  }));
};

export const getUserHistory = async (userId: number) => {
  if (!isStaticHost) {
    try {
      const res = await api.get(`/api/users/${userId}/history`);
      if (res.data?.history) return res.data.history;
    } catch {}
  }

  const movies = await loadMovies();
  const sample = movies.filter((m: any) => m.rating_count > 1000).slice(0, 10);
  return sample.map((m: any) => ({
    movie_idx: m.movie_idx,
    movieId: m.movieId,
    title: m.title,
    release_year: m.release_year,
    genres: m.genres,
    rating_mean: m.rating_mean,
  }));
};

export const getUserSaved = async (userId: number) => {
  if (!isStaticHost) {
    try {
      const res = await api.get(`/api/users/${userId}/saved`);
      if (res.data?.saved) return res.data.saved;
    } catch {}
  }

  const savedStr = localStorage.getItem(`cinepulse_saved_${userId}`);
  if (!savedStr) return [];
  const savedIds: number[] = JSON.parse(savedStr);
  const movies = await loadMovies();
  return movies
    .filter(
      (m: any) => savedIds.includes(m.movie_idx) || savedIds.includes(m.movieId)
    )
    .map((m: any) => ({
      movie_idx: m.movie_idx,
      movieId: m.movieId,
      title: m.title,
      release_year: m.release_year,
      genres: m.genres,
      rating_mean: m.rating_mean,
    }));
};

export const getUserInsights = async (userId: number): Promise<UserInsights> => {
  if (!isStaticHost) {
    try {
      const res = await api.get(`/api/users/${userId}/insights`);
      if (res.data) return res.data;
    } catch {}
  }

  const savedStr = localStorage.getItem(`cinepulse_saved_${userId}`);
  const likedStr = localStorage.getItem(`cinepulse_liked_${userId}`);
  const dismissedStr = localStorage.getItem(`cinepulse_dismissed_${userId}`);
  const savedCount = savedStr ? JSON.parse(savedStr).length : 0;
  const likedCount = likedStr ? JSON.parse(likedStr).length : 4;
  const dismissedCount = dismissedStr ? JSON.parse(dismissedStr).length : 0;

  return {
    user_idx: userId,
    total_interactions: 45 + likedCount + savedCount,
    active_likes_count: likedCount,
    active_saves_count: savedCount,
    dismissed_count: dismissedCount,
    top_affinity_genres: [
      { genre: 'Sci-Fi', affinity: 0.38 },
      { genre: 'Mystery', affinity: 0.24 },
      { genre: 'Thriller', affinity: 0.18 },
      { genre: 'Drama', affinity: 0.12 },
      { genre: 'Action', affinity: 0.08 },
    ],
    retrieval_sources_breakdown: {
      two_tower: 100,
      item_cf: 80,
      popularity: 20,
    },
    timing_breakdown_ms: {
      candidate_retrieval_ms: 6.2,
      lightgbm_ranking_ms: 10.4,
      mmr_reranking_ms: 3.1,
      total_pipeline_ms: 19.7,
    },
  };
};

export const getModelMetrics = async (): Promise<ModelEvaluation> => {
  if (!isStaticHost) {
    try {
      const res = await api.get('/api/metrics');
      if (res.data) return res.data;
    } catch {}
  }

  const res = await fetch(`${getBaseUrl()}demo-data/metrics.json`);
  return res.json();
};
