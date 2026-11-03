import React, { useState } from 'react';
import { SlidersHorizontal, RefreshCw, Sparkles, AlertCircle } from 'lucide-react';
import type { RecommendationFeed } from '../types';
import { HeroBanner } from '../components/HeroBanner';
import { MovieRail } from '../components/MovieRail';

interface HomePageProps {
  feed: RecommendationFeed | null;
  loading: boolean;
  onRefresh: (diversity?: number, novelty?: number) => void;
  onOpenDetail: (movieId: number) => void;
  onFeedback: (movieId: number, action: 'like' | 'dislike' | 'save' | 'dismiss') => void;
  activeProfileName: string;
}

export const HomePage: React.FC<HomePageProps> = ({
  feed,
  loading,
  onRefresh,
  onOpenDetail,
  onFeedback,
  activeProfileName,
}) => {
  const [diversity, setDiversity] = useState(0.65);
  const [novelty, setNovelty] = useState(0.15);
  const [showTuning, setShowTuning] = useState(false);

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Feed Sub-header / Tuning Controls */}
      <div className="flex flex-wrap items-center justify-between gap-4 mb-6 pb-4 border-b border-border">
        <div>
          <span className="text-[11px] font-mono tracking-wider text-muted uppercase">
            Curated Discovery Feed
          </span>
          <h2 className="font-display font-bold text-2xl text-white">
            Recommended for {activeProfileName}
          </h2>
        </div>

        <div className="flex items-center space-x-3">
          <button
            onClick={() => setShowTuning(!showTuning)}
            className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg border text-xs font-medium transition ${
              showTuning
                ? 'bg-surface-elevated text-white border-accent'
                : 'bg-surface text-gray-300 border-border hover:border-border-focus'
            }`}
          >
            <SlidersHorizontal className="w-3.5 h-3.5 text-accent" />
            <span>Tune MMR & Novelty</span>
          </button>

          <button
            onClick={() => onRefresh(diversity, novelty)}
            disabled={loading}
            className="flex items-center space-x-2 px-3.5 py-1.5 rounded-lg bg-surface hover:bg-surface-elevated border border-border text-xs text-white transition disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-accent' : 'text-muted'}`} />
            <span>Refresh Feed</span>
          </button>
        </div>
      </div>

      {/* Real-time MMR Tuning Panel */}
      {showTuning && (
        <div className="mb-8 p-5 rounded-xl bg-surface-elevated border border-border/80 shadow-lg">
          <div className="flex items-center space-x-2 text-xs font-semibold text-accent mb-4">
            <Sparkles className="w-4 h-4" />
            <span>Stage D: Dynamic MMR & Exploration Hyperparameter Controls</span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <div className="flex justify-between text-xs text-gray-300 mb-1">
                <span>Relevance vs. Diversity Balance (λ)</span>
                <span className="font-mono text-accent font-bold">{diversity.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={diversity}
                onChange={(e) => setDiversity(parseFloat(e.target.value))}
                className="w-full accent-amber-400 bg-surface h-1.5 rounded-lg cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-muted font-mono mt-1">
                <span>Max Diversity (0.1)</span>
                <span>Balanced (0.65)</span>
                <span>Pure Relevance (1.0)</span>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs text-gray-300 mb-1">
                <span>Novelty / Long-Tail Exploration Weight (β)</span>
                <span className="font-mono text-accent font-bold">{novelty.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="0.4"
                step="0.02"
                value={novelty}
                onChange={(e) => setNovelty(parseFloat(e.target.value))}
                className="w-full accent-amber-400 bg-surface h-1.5 rounded-lg cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-muted font-mono mt-1">
                <span>Mainstream Hits (0.0)</span>
                <span>Moderate Novelty (0.15)</span>
                <span>Deep Obscurities (0.4)</span>
              </div>
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-border flex justify-end">
            <button
              onClick={() => onRefresh(diversity, novelty)}
              className="px-4 py-1.5 rounded-lg bg-accent text-black font-semibold text-xs hover:bg-accent-hover transition shadow"
            >
              Apply Reranking & Update Feed
            </button>
          </div>
        </div>
      )}

      {/* Loading Skeleton */}
      {loading && !feed ? (
        <div className="py-24 text-center">
          <div className="w-10 h-10 border-2 border-accent border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p className="text-sm font-medium text-white">Generating Multi-Stage Recommendations...</p>
          <p className="text-xs text-muted mt-1">Querying Two-Tower FAISS Index • LightGBM Scoring • MMR Re-ranking</p>
        </div>
      ) : feed ? (
        <div>
          {/* Hero Featured Recommendation */}
          {feed.hero_featured && (
            <HeroBanner
              movie={feed.hero_featured}
              onOpenDetail={onOpenDetail}
              onFeedback={onFeedback}
            />
          )}

          {/* Categorized Movie Rails */}
          {Object.entries(feed.rails).map(([railTitle, items]) => (
            <MovieRail
              key={railTitle}
              title={railTitle}
              subtitle={
                railTitle.includes('Top Picks')
                  ? 'High neural embedding similarity and LightGBM predicted affinity'
                  : railTitle.includes('Because You Watched')
                  ? 'Collaborative filtering nearest neighbors based on co-rating patterns'
                  : railTitle.includes('Hidden Gems')
                  ? 'Unfamiliar and less popular films with high positive sentiment'
                  : 'Acclaimed films with widespread consensus'
              }
              movies={items}
              onOpenDetail={onOpenDetail}
              onFeedback={onFeedback}
            />
          ))}
        </div>
      ) : (
        <div className="p-12 text-center text-muted border border-dashed border-border rounded-xl">
          <AlertCircle className="w-8 h-8 text-accent mx-auto mb-2" />
          <p className="text-sm text-gray-300">Unable to load feed. Ensure backend service is running.</p>
        </div>
      )}
    </div>
  );
};
