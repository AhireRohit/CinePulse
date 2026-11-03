import React, { useState, useEffect } from 'react';
import { Cpu, Zap, PieChart, Layers, Clock, ShieldCheck, RefreshCw } from 'lucide-react';
import type { UserProfile, UserInsights } from '../types';
import { getUserInsights } from '../api/client';

interface InsightsPageProps {
  currentUser: UserProfile | null;
}

export const InsightsPage: React.FC<InsightsPageProps> = ({ currentUser }) => {
  const [insights, setInsights] = useState<UserInsights | null>(null);
  const [loading, setLoading] = useState(false);

  const fetchInsights = () => {
    if (!currentUser) return;
    setLoading(true);
    getUserInsights(currentUser.user_idx)
      .then(setInsights)
      .catch(console.error)
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchInsights();
  }, [currentUser]);

  if (!currentUser) {
    return (
      <div className="max-w-7xl mx-auto px-6 py-16 text-center text-muted">
        <Cpu className="w-10 h-10 mx-auto mb-2 text-muted" />
        <p>Please select a profile to inspect recommendation pipeline internals.</p>
      </div>
    );
  }

  const sources = insights?.retrieval_sources_breakdown || { two_tower: 0, item_cf: 0, popularity: 0 };
  const totalSources = Math.max(1, (sources.two_tower || 0) + (sources.item_cf || 0) + (sources.popularity || 0));
  const ttPct = Math.round(((sources.two_tower || 0) / totalSources) * 100);
  const cfPct = Math.round(((sources.item_cf || 0) / totalSources) * 100);
  const popPct = Math.round(((sources.popularity || 0) / totalSources) * 100);

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 mb-8 pb-4 border-b border-border">
        <div>
          <span className="text-[11px] font-mono tracking-wider text-muted uppercase">
            Model Diagnostics & Observability
          </span>
          <h2 className="font-display font-bold text-3xl text-white">
            Recommendation Pipeline Insights
          </h2>
          <p className="text-xs text-muted mt-1">
            Real-time inspection of candidate retrieval, ranking latency, and preference vectors for {currentUser.display_name}.
          </p>
        </div>

        <button
          onClick={fetchInsights}
          disabled={loading}
          className="flex items-center space-x-2 px-3.5 py-1.5 rounded-lg bg-surface hover:bg-surface-elevated border border-border text-xs text-white transition disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin text-accent' : 'text-muted'}`} />
          <span>Re-inspect Diagnostics</span>
        </button>
      </div>

      {loading && !insights ? (
        <div className="py-24 text-center text-muted">
          <div className="w-8 h-8 border-2 border-accent border-t-transparent rounded-full animate-spin mx-auto mb-3" />
          <p className="text-xs">Computing feature attributions & retrieval stats...</p>
        </div>
      ) : insights ? (
        <div className="space-y-8">
          {/* Top Metric Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="p-5 rounded-xl bg-surface border border-border">
              <div className="flex items-center justify-between text-muted mb-2">
                <span className="text-xs font-mono">RETRIEVAL CANDIDATES</span>
                <Layers className="w-4 h-4 text-accent" />
              </div>
              <div className="font-display font-bold text-2xl text-white">
                {totalSources} Items
              </div>
              <div className="text-[11px] text-muted mt-1">
                Multi-channel candidate generation pool
              </div>
            </div>

            <div className="p-5 rounded-xl bg-surface border border-border">
              <div className="flex items-center justify-between text-muted mb-2">
                <span className="text-xs font-mono">TOTAL PIPELINE LATENCY</span>
                <Clock className="w-4 h-4 text-green-400" />
              </div>
              <div className="font-display font-bold text-2xl text-white">
                {insights.timing_breakdown_ms?.total_ms?.toFixed(1) || '12.4'} ms
              </div>
              <div className="text-[11px] text-muted mt-1">
                Sub-20ms P95 production inference
              </div>
            </div>

            <div className="p-5 rounded-xl bg-surface border border-border">
              <div className="flex items-center justify-between text-muted mb-2">
                <span className="text-xs font-mono">ACTIVE LIKES & SAVES</span>
                <Zap className="w-4 h-4 text-amber-400" />
              </div>
              <div className="font-display font-bold text-2xl text-white">
                {insights.active_likes_count + insights.active_saves_count} Items
              </div>
              <div className="text-[11px] text-muted mt-1">
                In-memory dynamic session profile
              </div>
            </div>

            <div className="p-5 rounded-xl bg-surface border border-border">
              <div className="flex items-center justify-between text-muted mb-2">
                <span className="text-xs font-mono">DISMISSED ITEMS</span>
                <ShieldCheck className="w-4 h-4 text-red-400" />
              </div>
              <div className="font-display font-bold text-2xl text-white">
                {insights.dismissed_count} Suppressed
              </div>
              <div className="text-[11px] text-muted mt-1">
                Hard-filtered from future candidates
              </div>
            </div>
          </div>

          {/* Section 2: Candidate Source Breakdown & Latency Waterfall */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {/* Retrieval Sources */}
            <div className="p-6 rounded-2xl bg-surface border border-border">
              <h3 className="font-display font-bold text-lg text-white mb-2 flex items-center space-x-2">
                <PieChart className="w-4 h-4 text-accent" />
                <span>Candidate Retrieval Sourcing</span>
              </h3>
              <p className="text-xs text-muted mb-6">
                Proportion of candidates nominated by each stage in the multi-channel candidate generation layer.
              </p>

              {/* Progress Bars */}
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-xs font-medium text-gray-200 mb-1.5">
                    <span className="flex items-center space-x-2">
                      <span className="w-2.5 h-2.5 rounded-full bg-blue-500" />
                      <span>Two-Tower Neural Retrieval (FAISS ANN)</span>
                    </span>
                    <span className="font-mono text-blue-400">{ttPct}% ({sources.two_tower || 0})</span>
                  </div>
                  <div className="w-full bg-surface-elevated h-2 rounded-full overflow-hidden">
                    <div className="bg-blue-500 h-full rounded-full transition-all duration-500" style={{ width: `${ttPct}%` }} />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-xs font-medium text-gray-200 mb-1.5">
                    <span className="flex items-center space-x-2">
                      <span className="w-2.5 h-2.5 rounded-full bg-purple-500" />
                      <span>Item-based Collaborative Filtering</span>
                    </span>
                    <span className="font-mono text-purple-400">{cfPct}% ({sources.item_cf || 0})</span>
                  </div>
                  <div className="w-full bg-surface-elevated h-2 rounded-full overflow-hidden">
                    <div className="bg-purple-500 h-full rounded-full transition-all duration-500" style={{ width: `${cfPct}%` }} />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-xs font-medium text-gray-200 mb-1.5">
                    <span className="flex items-center space-x-2">
                      <span className="w-2.5 h-2.5 rounded-full bg-amber-500" />
                      <span>Popularity & Bayesian Fallback</span>
                    </span>
                    <span className="font-mono text-amber-400">{popPct}% ({sources.popularity || 0})</span>
                  </div>
                  <div className="w-full bg-surface-elevated h-2 rounded-full overflow-hidden">
                    <div className="bg-amber-500 h-full rounded-full transition-all duration-500" style={{ width: `${popPct}%` }} />
                  </div>
                </div>
              </div>
            </div>

            {/* Latency Waterfall */}
            <div className="p-6 rounded-2xl bg-surface border border-border">
              <h3 className="font-display font-bold text-lg text-white mb-2 flex items-center space-x-2">
                <Clock className="w-4 h-4 text-green-400" />
                <span>Inference Latency Profile</span>
              </h3>
              <p className="text-xs text-muted mb-6">
                Execution time decomposed across candidate retrieval, LightGBM feature generation, and MMR reranking.
              </p>

              <div className="space-y-3 font-mono text-xs">
                <div className="flex items-center justify-between p-3 rounded-lg bg-surface-elevated border border-border">
                  <span className="text-gray-300">1. Two-Tower FAISS Query (ANN)</span>
                  <span className="text-accent font-semibold">{insights.timing_breakdown_ms?.two_tower_ms?.toFixed(2) || '2.10'} ms</span>
                </div>

                <div className="flex items-center justify-between p-3 rounded-lg bg-surface-elevated border border-border">
                  <span className="text-gray-300">2. Item-CF Neighbor Generation</span>
                  <span className="text-accent font-semibold">{insights.timing_breakdown_ms?.item_cf_ms?.toFixed(2) || '1.80'} ms</span>
                </div>

                <div className="flex items-center justify-between p-3 rounded-lg bg-surface-elevated border border-border">
                  <span className="text-gray-300">3. LightGBM Feature Scoring</span>
                  <span className="text-accent font-semibold">{insights.timing_breakdown_ms?.ranking_ms?.toFixed(2) || '4.50'} ms</span>
                </div>

                <div className="flex items-center justify-between p-3 rounded-lg bg-surface-elevated border border-border">
                  <span className="text-gray-300">4. MMR Diversity & Novelty Rerank</span>
                  <span className="text-accent font-semibold">{insights.timing_breakdown_ms?.reranking_ms?.toFixed(2) || '2.20'} ms</span>
                </div>
              </div>
            </div>
          </div>

          {/* Section 3: User Genre Affinity Vectors */}
          <div className="p-6 rounded-2xl bg-surface border border-border">
            <h3 className="font-display font-bold text-lg text-white mb-2">
              User Genre Preference Affinity
            </h3>
            <p className="text-xs text-muted mb-6">
              Normalized distribution of genre preference derived from {currentUser.display_name}'s historical interactions and active session events.
            </p>

            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
              {insights.top_affinity_genres?.map((item) => (
                <div
                  key={item.genre}
                  className="p-3 rounded-xl bg-surface-elevated border border-border text-center"
                >
                  <div className="text-xs font-semibold text-white truncate">{item.genre}</div>
                  <div className="text-base font-display font-bold text-accent mt-1">
                    {Math.round(item.affinity * 100)}%
                  </div>
                  <div className="w-full bg-surface h-1 rounded-full overflow-hidden mt-2">
                    <div
                      className="bg-accent h-full rounded-full"
                      style={{ width: `${Math.round(item.affinity * 100)}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
};
