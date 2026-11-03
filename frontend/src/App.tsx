import React, { useState, useEffect } from 'react';
import type { UserProfile, RecommendationFeed } from './types';
import { getUsers, getRecommendations, postFeedback } from './api/client';
import { Navbar } from './components/Navbar';
import { HomePage } from './pages/HomePage';
import { ExplorePage } from './pages/ExplorePage';
import { LibraryPage } from './pages/LibraryPage';
import { InsightsPage } from './pages/InsightsPage';
import { EvaluationPage } from './pages/EvaluationPage';
import { MovieDetailModal } from './components/MovieDetailModal';
import { OnboardingModal } from './components/OnboardingModal';

export const App: React.FC = () => {
  const [currentTab, setCurrentTab] = useState('feed');
  const [users, setUsers] = useState<UserProfile[]>([]);
  const [currentUser, setCurrentUser] = useState<UserProfile | null>(null);
  const [feed, setFeed] = useState<RecommendationFeed | null>(null);
  const [loadingFeed, setLoadingFeed] = useState(false);
  const [selectedMovieId, setSelectedMovieId] = useState<number | null>(null);
  const [onboardingOpen, setOnboardingOpen] = useState(false);
  const [sessionEventCount, setSessionEventCount] = useState(0);
  const [feedbackNotification, setFeedbackNotification] = useState<string | null>(null);

  // Load user profiles on mount
  useEffect(() => {
    getUsers()
      .then((data) => {
        setUsers(data);
        if (data.length > 0) {
          setCurrentUser(data[0]);
        }
      })
      .catch((err) => console.error('Failed to load user profiles:', err));
  }, []);

  // Fetch recommendations when current user changes
  const loadFeed = (diversity = 0.65, novelty = 0.15) => {
    if (!currentUser) return;
    setLoadingFeed(true);
    getRecommendations(currentUser.user_idx, { diversity, novelty })
      .then((data) => setFeed(data))
      .catch((err) => console.error('Failed to fetch recommendations:', err))
      .finally(() => setLoadingFeed(false));
  };

  useEffect(() => {
    if (currentUser) {
      loadFeed();
    }
  }, [currentUser]);

  // Handle feedback (Like, Dislike, Save, Dismiss, Rate)
  const handleFeedback = async (
    movieId: number,
    action: 'like' | 'dislike' | 'save' | 'dismiss' | 'rate',
    rating?: number
  ) => {
    if (!currentUser) return;
    try {
      await postFeedback(currentUser.user_idx, movieId, action, rating);
      setSessionEventCount((prev) => prev + 1);

      const msg =
        action === 'like'
          ? 'Liked! Recommendations boosted for similar titles.'
          : action === 'dislike'
          ? 'Disliked! Item suppressed and genre weights reduced.'
          : action === 'save'
          ? 'Saved to your personal Watchlist.'
          : action === 'rate'
          ? `Rated ${rating}★! Session vector updated.`
          : 'Dismissed! Hidden from future candidates.';

      setFeedbackNotification(msg);
      setTimeout(() => setFeedbackNotification(null), 3500);

      // Refresh recommendations to demonstrate real-time dynamic adaptation
      loadFeed();
    } catch (err) {
      console.error('Failed to submit feedback:', err);
    }
  };

  return (
    <div className="min-h-screen bg-background text-gray-100 flex flex-col font-sans">
      {/* Navigation Bar */}
      <Navbar
        currentTab={currentTab}
        setCurrentTab={setCurrentTab}
        users={users}
        currentUser={currentUser}
        onSelectUser={(u) => {
          setCurrentUser(u);
          setSessionEventCount(0);
        }}
        onOpenOnboarding={() => setOnboardingOpen(true)}
        sessionEventCount={sessionEventCount}
      />

      {/* Floating Session Real-time Adaptation Toast */}
      {feedbackNotification && (
        <div className="fixed bottom-6 right-6 z-50 px-4 py-3 rounded-xl bg-surface-elevated border border-accent/40 shadow-2xl flex items-center space-x-3 text-xs text-white animate-fade-in">
          <span className="w-2 h-2 rounded-full bg-accent animate-ping" />
          <span>{feedbackNotification}</span>
        </div>
      )}

      {/* Main Content Pages */}
      <main className="flex-1">
        {currentTab === 'feed' && (
          <HomePage
            feed={feed}
            loading={loadingFeed}
            onRefresh={loadFeed}
            onOpenDetail={(id) => setSelectedMovieId(id)}
            onFeedback={handleFeedback}
            activeProfileName={currentUser?.display_name || 'You'}
          />
        )}

        {currentTab === 'explore' && (
          <ExplorePage
            onOpenDetail={(id) => setSelectedMovieId(id)}
            onFeedback={handleFeedback}
          />
        )}

        {currentTab === 'library' && (
          <LibraryPage
            currentUser={currentUser}
            onOpenDetail={(id) => setSelectedMovieId(id)}
            onFeedback={handleFeedback}
          />
        )}

        {currentTab === 'insights' && (
          <InsightsPage currentUser={currentUser} />
        )}

        {currentTab === 'evaluation' && (
          <EvaluationPage />
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-border py-8 text-center text-xs text-muted">
        <div className="max-w-7xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center space-x-2">
            <span className="font-display font-semibold text-white">CINEPULSE</span>
            <span>— Production Multi-Stage Recommendation Platform</span>
          </div>
          <div className="text-[11px] font-mono text-muted">
            GroupLens MovieLens 1M • Two-Tower Neural Retrieval • LightGBM LambdaRank • FAISS ANN
          </div>
        </div>
      </footer>

      {/* Movie Details Modal */}
      <MovieDetailModal
        movieId={selectedMovieId}
        onClose={() => setSelectedMovieId(null)}
        onFeedback={handleFeedback}
      />

      {/* Onboarding Modal */}
      {onboardingOpen && (
        <OnboardingModal
          onClose={() => setOnboardingOpen(false)}
          onUserCreated={(newUser) => {
            setUsers((prev) => [...prev, newUser]);
            setCurrentUser(newUser);
            setSessionEventCount(0);
          }}
        />
      )}
    </div>
  );
};
export default App;
