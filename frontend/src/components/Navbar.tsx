import React, { useState } from 'react';
import { Film, Compass, Bookmark, Cpu, BarChart2, Plus, Check, Zap, Sparkles } from 'lucide-react';
import type { UserProfile } from '../types';
import { GlobalSearchBar } from './GlobalSearchBar';

interface NavbarProps {
  currentTab: string;
  setCurrentTab: (tab: string) => void;
  users: UserProfile[];
  currentUser: UserProfile | null;
  onSelectUser: (user: UserProfile) => void;
  onOpenOnboarding: () => void;
  sessionEventCount: number;
  onOpenDetail: (movieId: number) => void;
  onFeedback: (
    movieId: number,
    action: 'like' | 'dislike' | 'save' | 'dismiss' | 'rate',
    rating?: number
  ) => void;
  onOpenSessionDrawer: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentTab,
  setCurrentTab,
  users,
  currentUser,
  onSelectUser,
  onOpenOnboarding,
  sessionEventCount,
  onOpenDetail,
  onFeedback,
  onOpenSessionDrawer,
}) => {
  const [profileDropdownOpen, setProfileDropdownOpen] = useState(false);

  const navItems = [
    { id: 'feed', label: 'Discover', icon: Film },
    { id: 'explore', label: 'Explore', icon: Compass },
    { id: 'library', label: 'Library', icon: Bookmark },
    { id: 'insights', label: 'Insights', icon: Cpu },
    { id: 'evaluation', label: 'Benchmarks', icon: BarChart2 },
  ];

  return (
    <header className="sticky top-0 z-40 bg-background/95 backdrop-blur-md border-b border-border transition-all">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between gap-2 sm:gap-4">
        {/* Left: Brand Logo */}
        <div className="flex items-center space-x-6 sm:space-x-8 flex-shrink-0">
          <div
            onClick={() => setCurrentTab('feed')}
            className="flex items-center space-x-2.5 cursor-pointer group"
          >
            <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-amber-400 to-amber-600 flex items-center justify-center shadow-lg shadow-amber-500/20 group-hover:scale-105 transition-transform">
              <Film className="w-4 h-4 text-black font-extrabold" />
            </div>
            <div>
              <span className="font-display font-extrabold tracking-wider text-base text-white">
                CINE<span className="text-accent">PULSE</span>
              </span>
              <span className="hidden sm:block text-[9px] tracking-widest text-muted uppercase font-mono">
                RECOMMENDER
              </span>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="hidden lg:flex items-center space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const active = currentTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setCurrentTab(item.id)}
                  className={`flex items-center space-x-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                    active
                      ? 'bg-surface-elevated text-white border border-border-focus shadow-sm'
                      : 'text-muted hover:text-white hover:bg-surface'
                  }`}
                >
                  <Icon className={`w-3.5 h-3.5 ${active ? 'text-accent' : 'text-muted'}`} />
                  <span>{item.label}</span>
                </button>
              );
            })}
          </nav>
        </div>

        {/* Center: Global Search Bar */}
        <div className="flex-1 max-w-xs sm:max-w-md mx-2 flex justify-center">
          <GlobalSearchBar
            onOpenDetail={onOpenDetail}
            onFeedback={onFeedback}
          />
        </div>

        {/* Right: Live Session Telemetry & User Profile Switcher */}
        <div className="flex items-center space-x-2 sm:space-x-3 flex-shrink-0">
          {/* Real-time Session Activity Pill */}
          <button
            onClick={onOpenSessionDrawer}
            title="Inspect Real-Time Session Adaptation"
            className={`flex items-center space-x-1.5 px-2.5 py-1.5 rounded-lg border text-xs font-medium transition ${
              sessionEventCount > 0
                ? 'bg-amber-500/15 border-amber-500/40 text-accent animate-pulse'
                : 'bg-surface border-border text-muted hover:text-white hover:border-border-focus'
            }`}
          >
            <Zap className="w-3.5 h-3.5 text-accent" />
            <span className="hidden sm:inline font-mono text-[11px]">
              {sessionEventCount > 0 ? `${sessionEventCount} Session Events` : 'Session Active'}
            </span>
          </button>

          {/* Profile Switcher Dropdown */}
          <div className="relative">
            <button
              onClick={() => setProfileDropdownOpen(!profileDropdownOpen)}
              className="flex items-center space-x-2 py-1 px-2.5 rounded-xl bg-surface border border-border hover:border-accent/40 transition shadow-sm"
            >
              <div
                className="w-6 h-6 rounded-lg flex items-center justify-center text-xs font-bold text-white shadow"
                style={{ backgroundColor: currentUser?.avatar_color || '#3B82F6' }}
              >
                {currentUser?.display_name.charAt(0) || 'U'}
              </div>
              <span className="text-xs font-semibold text-gray-200 hidden md:inline">
                {currentUser?.display_name || 'Select Profile'}
              </span>
            </button>

            {profileDropdownOpen && (
              <div className="absolute right-0 mt-2 w-80 bg-surface-elevated border border-border rounded-2xl shadow-2xl p-2.5 z-50 animate-fade-in">
                <div className="flex items-center justify-between px-3 py-2 border-b border-border">
                  <span className="text-[10px] font-mono tracking-wider text-muted uppercase">
                    Persona & Profiles
                  </span>
                  <span className="text-[10px] text-accent flex items-center space-x-1">
                    <Sparkles className="w-3 h-3" />
                    <span>Real Data Mapped</span>
                  </span>
                </div>

                <div className="py-1 max-h-72 overflow-y-auto divide-y divide-border/20">
                  {users.map((u) => {
                    const isSelected = currentUser?.user_idx === u.user_idx;
                    return (
                      <div
                        key={u.user_idx}
                        onClick={() => {
                          onSelectUser(u);
                          setProfileDropdownOpen(false);
                        }}
                        className={`flex items-center justify-between px-3 py-2.5 rounded-xl cursor-pointer transition ${
                          isSelected ? 'bg-surface border border-accent/30' : 'hover:bg-surface'
                        }`}
                      >
                        <div className="flex items-center space-x-3">
                          <div
                            className="w-8 h-8 rounded-xl flex items-center justify-center text-xs font-bold text-white shadow-sm flex-shrink-0"
                            style={{ backgroundColor: u.avatar_color }}
                          >
                            {u.display_name.charAt(0)}
                          </div>
                          <div className="min-w-0">
                            <div className="text-xs font-semibold text-white truncate">
                              {u.display_name}
                            </div>
                            <div className="text-[10px] text-muted truncate">
                              {u.preferred_genres?.slice(0, 3).join(' • ') || 'Movie Fan'}
                            </div>
                          </div>
                        </div>
                        {isSelected && <Check className="w-4 h-4 text-accent flex-shrink-0" />}
                      </div>
                    );
                  })}
                </div>

                <div className="pt-2 border-t border-border mt-1">
                  <button
                    onClick={() => {
                      setProfileDropdownOpen(false);
                      onOpenOnboarding();
                    }}
                    className="w-full flex items-center justify-center space-x-2 py-2.5 px-3 rounded-xl bg-accent text-black font-semibold text-xs hover:bg-accent-hover transition shadow"
                  >
                    <Plus className="w-4 h-4" />
                    <span>Create Cold-Start Profile</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Mobile Navigation Bar */}
      <div className="lg:hidden flex items-center justify-around px-2 py-2 border-t border-border/80 bg-surface/50 text-xs">
        {navItems.map((item) => {
          const Icon = item.icon;
          const active = currentTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setCurrentTab(item.id)}
              className={`flex flex-col items-center py-1 px-2 rounded-md ${
                active ? 'text-accent' : 'text-muted hover:text-white'
              }`}
            >
              <Icon className="w-4 h-4 mb-0.5" />
              <span className="text-[10px]">{item.label}</span>
            </button>
          );
        })}
      </div>
    </header>
  );
};
