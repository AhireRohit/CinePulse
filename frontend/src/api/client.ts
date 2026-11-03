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
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getHealth = async () => {
  const res = await api.get('/health');
  return res.data;
};

export const getUsers = async (): Promise<UserProfile[]> => {
  const res = await api.get('/api/users');
  return res.data;
};

export const createUser = async (data: {
  username: string;
  display_name: string;
  preferred_genres: string[];
  favorite_movies: number[];
  avatar_color?: string;
  bio?: string;
}): Promise<UserProfile> => {
  const res = await api.post('/api/users', data);
  return res.data;
};

export const getRecommendations = async (
  userId: number,
  params?: { k?: number; diversity?: number; novelty?: number }
): Promise<RecommendationFeed> => {
  const res = await api.get(`/api/recommendations/${userId}`, { params });
  return res.data;
};

export const postFeedback = async (
  userId: number,
  movieId: number,
  action: 'like' | 'dislike' | 'save' | 'dismiss' | 'rate',
  rating?: number
) => {
  const res = await api.post('/api/feedback', {
    user_idx: userId,
    movie_idx: movieId,
    action,
    rating,
  });
  return res.data;
};

export const getMovie = async (movieId: number): Promise<MovieDetail> => {
  const res = await api.get(`/api/movies/${movieId}`);
  return res.data;
};

export const searchMovies = async (q: string): Promise<RecommendationItem[]> => {
  const res = await api.get('/api/search', { params: { q } });
  return res.data.results;
};

export const getUserHistory = async (userId: number) => {
  const res = await api.get(`/api/users/${userId}/history`);
  return res.data.history;
};

export const getUserSaved = async (userId: number) => {
  const res = await api.get(`/api/users/${userId}/saved`);
  return res.data.saved;
};

export const getUserInsights = async (userId: number): Promise<UserInsights> => {
  const res = await api.get(`/api/users/${userId}/insights`);
  return res.data;
};

export const getModelMetrics = async (): Promise<ModelEvaluation> => {
  const res = await api.get('/api/metrics');
  return res.data;
};
