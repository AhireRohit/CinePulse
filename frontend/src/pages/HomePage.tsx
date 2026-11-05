import React, { useState } from 'react';
import { SlidersHorizontal, RefreshCw, Sparkles, AlertCircle, Compass, Zap } from 'lucide-react';
import type { RecommendationFeed } from '../types';
import { HeroBanner } from '../components/HeroBanner';
import { MovieRail } from '../components/MovieRail';
import { isStaticHost } from '../api/client';

interface HomePageProps {
  feed: RecommendationFeed | null;
  loading: boolean;
  onRefresh: (diversity?: number, novelty?: number) => void;
  onOpenDetail: (movieId: number) => void;
  onFeedback: (
    movieId: number,
    action: 'like' | 'dislike' | 'save' | 'dismiss' | 'rate',
    rating?: number
  ) => void;
  activeProfileName: string;
}

const DISCOVERY_VIBES = [
  { id: 'all', label: 'All Curations', icon: Compass },
  { id: 'scifi', label: 'Mind-Bending', icon: Sparkles, target: 'Sci-Fi' },
  { id: 'crime', label: 'Dark & Gritty', icon: Zap, target: 'Crime' },
  { id: 'gems', label: 'Hidden Gems', icon: Sparkles, target: 'Hidden Gems' },
  { id: 'popular', label: 'Masterpieces', icon: Compass, target: 'Top Rated' },
];

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
  const [selectedVibe, setSelectedVibe] = useState('all');

  const applyPreset = (preset: 'discovery' | 'balanced' | 'relevance') => {
    if (preset === 'discovery') {
      setDiversity(0.35);
      setNovelty(0.30);
      onRefresh(0.35, 0.30);
    } else if (preset === 'balanced') {
      setDiversity(0.65);
      setNovelty(0.15);
      onRefresh(0.65, 0.15);
    } else {
      setDiversity(0.95);
      setNovelty(0.02);
      onRefresh(0.95, 0.02);
    }
  };

  // Filter rails if a specific vibe is chosen
  const filteredRails = feed
    ? Object.entries(feed.rails).filter(([title]) => {
        if (selectedVibe === 'all') return true;
        const target = DISCOVERY_VIBES.find((v) => v.id === selectedVibe)?.target;
        if (!target) return true;
        return title.toLowerCase().includes(target.toLowerCase());
      })
    : [];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
      {/* Feed Sub-header / Controls */}
      <div className="flex flex-wrap items-center justify-between gap-4 mb-6 pb-4 border-b border-border/80">
        <div>
          <div className="flex items-center space-x-2 text-[11px] font-mono tracking-wider text-muted uppercase">
            <span className={`w-2 h-2 rounded-full ${isStaticHost ? 'bg-amber-400' : 'bg-emerald-400'}`} />
            <span>
              {isStaticHost
                ? 'Precomputed V2 Model Showcase (GitHub Pages)'
                : 'Live CinePulse V2 Serving Pipeline Active'}
            </span>
          </div>
          <h2 className="font-display font-extrabold text-2xl sm:text-3xl text-white tracking-tight mt-0.5">
            Curated For {activeProfileName}
          </h2>
        </div>

        <div className="flex items-center space-x-2.5">
          <button
            onClick={() => setShowTuning(!showTuning)}
            className={`flex items-center space-x-2 px-3.5 py-2 rounded-xl border text-xs font-semibold transition ${
              showTuning
                ? 'bg-accent text-black border-accent shadow-md shadow-accent/20'
                : 'bg-surface text-gray-200 border-border hover:border-accent/40'
            }`}
          >
            <SlidersHorizontal className="w-3.5 h-3.5" />
            <span>Tune Diversity & Novelty</span>
          </button>

          <button
            onClick={() => onRefresh(diversity, novelty)}
            disabled={loading}
            className="flex items-center space-x-2 px-4 py-2 rounded-xl bg-surface hover:bg-surface-elevated border border-border text-xs text-white font-medium transition disabled:opacity-50 shadow-sm"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-accent' : 'text-muted'}`} />
            <span>Recalibrate Feed</span>
          </button>
        </div>
      </div>

      {/* Discovery Mood / Vibe Filter Pills */}
      <div className="flex items-center space-x-2 overflow-x-auto pb-4 mb-6 scrollbar-none">
        {DISCOVERY_VIBES.map((vibe) => {
          const Icon = vibe.icon;
          const active = selectedVibe === vibe.id;
          return (
            <button
              key={vibe.id}
              onClick={() => setSelectedVibe(vibe.id)}
              className={`flex-shrink-0 flex items-center space-x-1.5 px-3.5 py-1.5 rounded-xl text-xs font-semibold transition ${
                active
                  ? 'bg-surface-elevated text-accent border border-accent/40 shadow-sm'
                  : 'bg-surface hover:bg-surface-elevated text-muted hover:text-white border border-border'
              }`}
            >
              <Icon className="w-3 h-3" />
              <span>{vibe.label}</span>
            </button>
          );
        })}
      </div>

      {/* Real-time MMR Tuning Panel */}
      {showTuning && (
        <div className="mb-8 p-6 rounded-2xl bg-surface-elevated border border-accent/30 shadow-2xl animate-fade-in">
          {isStaticHost && (
            <div className="mb-4 p-3 rounded-xl bg-amber-500/10 border border-amber-500/30 flex items-start space-x-2 text-xs text-amber-200">
              <AlertCircle className="w-4 h-4 text-accent flex-shrink-0 mt-0.5" />
              <div>
                <span className="font-semibold text-white">Showcase Notice: </span>
                Dynamic MMR slider tuning requires the local Python FastAPI backend with active in-process FAISS/LightGBM re-scoring. This static GitHub Pages deployment serves authentic precomputed V2 production recommendations (&lambda;=0.95, &beta;=0.01).
              </div>
            </div>
          )}

          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center space-x-2 text-xs font-bold text-accent">
              <Sparkles className="w-4 h-4" />
              <span>Stage D: Dynamic Maximal Marginal Relevance (MMR) Calibration</span>
            </div>

            {/* Quick Presets */}
            <div className="flex items-center space-x-2">
              <span className="text-[10px] font-mono text-muted uppercase">Presets:</span>
              <button
                onClick={() => applyPreset('relevance')}
                className="px-2 py-0.5 rounded-md bg-surface hover:bg-surface-hover border border-border text-[11px] text-gray-300"
              >
                Pure Accuracy
              </button>
              <button
                onClick={() => applyPreset('balanced')}
                className="px-2 py-0.5 rounded-md bg-surface hover:bg-surface-hover border border-border text-[11px] text-accent font-semibold"
              >
                Editorial Balanced
              </button>
              <button
                onClick={() => applyPreset('discovery')}
                className="px-2 py-0.5 rounded-md bg-surface hover:bg-surface-hover border border-border text-[11px] text-purple-300"
              >
                Deep Discovery
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div>
              <div className="flex justify-between text-xs text-gray-200 mb-2">
                <span className="font-medium">Relevance vs. Diversity Balance (λ)</span>
                <span className="font-mono text-accent font-bold text-sm">{diversity.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.05"
                value={diversity}
                onChange={(e) => setDiversity(parseFloat(e.target.value))}
                className="w-full accent-amber-400 bg-surface h-2 rounded-lg cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-muted font-mono mt-1.5">
                <span>Max Diversity (0.1)</span>
                <span className="text-accent font-semibold">Standard Balanced (0.65)</span>
                <span>Pure Relevance (1.0)</span>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs text-gray-200 mb-2">
                <span className="font-medium">Novelty / Long-Tail Exploration Weight (β)</span>
                <span className="font-mono text-accent font-bold text-sm">{novelty.toFixed(2)}</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="0.4"
                step="0.02"
                value={novelty}
                onChange={(e) => setNovelty(parseFloat(e.target.value))}
                className="w-full accent-amber-400 bg-surface h-2 rounded-lg cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-muted font-mono mt-1.5">
                <span>Blockbusters Only (0.0)</span>
                <span className="text-accent font-semibold">Balanced Novelty (0.15)</span>
                <span>Deep Obscurities (0.4)</span>
              </div>
            </div>
          </div>

          <div className="mt-5 pt-4 border-t border-border flex items-center justify-between">
            <span className="text-[11px] text-muted">
              Live updates apply Maximal Marginal Relevance greedy reranking over LightGBM ranked candidates.
            </span>
            <button
              onClick={() => onRefresh(diversity, novelty)}
              className="px-5 py-2 rounded-xl bg-accent text-black font-bold text-xs hover:bg-accent-hover transition shadow-md shadow-accent/20"
            >
              Re-score Rails Now
            </button>
          </div>
        </div>
      )}

      {/* Loading Skeleton */}
      {loading && !feed ? (
        <div className="py-28 text-center">
          <div className="w-12 h-12 border-3 border-accent border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <h4 className="text-base font-semibold text-white">Generating Personalized Recommendations...</h4>
          <p className="text-xs text-muted mt-1 font-mono">
            Querying Two-Tower FAISS Index • LightGBM Cross-Scoring • MMR Re-ranking
          </p>
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
          {(filteredRails.length > 0 ? filteredRails : Object.entries(feed.rails)).map(
            ([railTitle, items]) => (
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
            )
          )}
        </div>
      ) : (
        <div className="p-14 text-center text-muted border border-dashed border-border rounded-2xl bg-surface/40">
          <AlertCircle className="w-9 h-9 text-accent mx-auto mb-3" />
          <p className="text-sm text-gray-200 font-medium">Unable to load feed. Ensure backend service is running.</p>
        </div>
      )}
    </div>
  );
};
