export interface UserProfile {
  user_idx: number;
  username: string;
  display_name: string;
  avatar_color: string;
  bio?: string;
  preferred_genres: string[];
}

export interface RecommendationItem {
  rank: number;
  movie_idx: number;
  movieId: number;
  title: string;
  full_title: string;
  release_year: number;
  genres: string[];
  rating_mean: number;
  rating_count: number;
  final_score: number;
  relevance_score: number;
  diversity_penalty: number;
  novelty_score: number;
  retrieval_source: string;
  reason: string;
  poster_url?: string;
  backdrop_url?: string;
}

export interface RecommendationFeed {
  user_idx: number;
  hero_featured?: RecommendationItem;
  rails: Record<string, RecommendationItem[]>;
  total_candidates: number;
  retrieval_sources_breakdown: Record<string, number>;
  timings_ms: Record<string, number>;
}

export interface MovieDetail {
  movie_idx: number;
  movieId: number;
  title: string;
  full_title: string;
  release_year: number;
  genres: string[];
  rating_mean: number;
  rating_count: number;
  bayesian_rating: number;
  overview?: string;
  poster_url?: string;
  backdrop_url?: string;
  similar_movies: RecommendationItem[];
}

export interface UserInsights {
  user_idx: number;
  total_interactions: number;
  active_likes_count: number;
  active_saves_count: number;
  dismissed_count: number;
  top_affinity_genres: { genre: string; affinity: number }[];
  retrieval_sources_breakdown: Record<string, number>;
  timing_breakdown_ms: Record<string, number>;
}

export interface ModelMetrics {
  'recall@20': number;
  'recall@100': number;
  'ndcg@10': number;
  'mrr@10': number;
  'catalog_coverage@20': number;
  'catalog_coverage@100': number;
  'intra_list_diversity': number;
  'novelty_bits': number;
  'latency_p50_ms': number;
  'latency_p95_ms': number;
}

export interface ModelEvaluation {
  dataset_name: string;
  num_users: number;
  num_movies: number;
  test_samples: number;
  models: Record<string, ModelMetrics>;
}
