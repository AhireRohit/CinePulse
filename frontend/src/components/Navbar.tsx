import React, { useState } from 'react';
import { Film, Compass, Bookmark, Cpu, BarChart2, Plus, Check } from 'lucide-react';
import type { UserProfile } from '../types';

interface NavbarProps {
  currentTab: string;
  setCurrentTab: (tab: string) => void;
  users: UserProfile[];
  currentUser: UserProfile | null;
  onSelectUser: (user: UserProfile) => void;
  onOpenOnboarding: () => void;
  sessionEventCount: number;
}

export const Navbar: React.FC<NavbarProps> = ({
  currentTab,
  setCurrentTab,
  users,
  currentUser,
  onSelectUser,
  onOpenOnboarding,
  sessionEventCount,
}) => {
  const [profileDropdownOpen, setProfileDropdownOpen] = useState(false);

  const navItems = [
    { id: 'feed', label: 'Discover', icon: Film },
    { id: 'explore', label: 'Explore', icon: Compass },
    { id: 'library', label: 'My Library', icon: Bookmark },
    { id: 'insights', label: 'Insights', icon: Cpu },
    { id: 'evaluation', label: 'Evaluation', icon: BarChart2 },
  ];

  return (
    <header className="sticky top-0 z-50 bg-background/90 backdrop-blur-md border-b border-border">
      <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center space-x-8">
          <div
            onClick={() => setCurrentTab('feed')}
            className="flex items-center space-x-2.5 cursor-pointer group"
          >
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-amber-400 to-amber-600 flex items-center justify-center shadow-lg shadow-amber-500/10">
              <Film className="w-4 h-4 text-black font-bold" />
            </div>
            <div>
              <span className="font-display font-semibold tracking-wider text-base text-white">
                CINE<span className="text-accent">PULSE</span>
              </span>
              <span className="block text-[10px] tracking-widest text-muted uppercase font-mono">
                RECOMMENDER
              </span>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="hidden md:flex items-center space-x-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              const active = currentTab === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setCurrentTab(item.id)}
                  className={`flex items-center space-x-2 px-3.5 py-1.5 rounded-md text-xs font-medium transition-all ${
                    active
                      ? 'bg-surface-elevated text-white border border-border-focus'
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

        {/* User Profile Switcher */}
        <div className="flex items-center space-x-4">
          {sessionEventCount > 0 && (
            <div className="hidden sm:flex items-center space-x-2 px-2.5 py-1 rounded-full bg-surface-elevated border border-border text-[11px] text-accent animate-pulse">
              <span className="w-1.5 h-1.5 rounded-full bg-accent" />
              <span>{sessionEventCount} Session Updates</span>
            </div>
          )}

          <div className="relative">
            <button
              onClick={() => setProfileDropdownOpen(!profileDropdownOpen)}
              className="flex items-center space-x-2.5 py-1 px-2.5 rounded-lg bg-surface border border-border hover:border-border-focus transition"
            >
              <div
                className="w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold text-white shadow"
                style={{ backgroundColor: currentUser?.avatar_color || '#3B82F6' }}
              >
                {currentUser?.display_name.charAt(0) || 'U'}
              </div>
              <span className="text-xs font-medium text-gray-200 hidden sm:inline">
                {currentUser?.display_name || 'Select Profile'}
              </span>
            </button>

            {profileDropdownOpen && (
              <div className="absolute right-0 mt-2 w-72 bg-surface-elevated border border-border rounded-xl shadow-2xl p-2 z-50">
                <div className="px-3 py-2 text-[11px] font-mono tracking-wider text-muted uppercase border-b border-border">
                  Demo & Local Profiles
                </div>
                <div className="py-1 max-h-64 overflow-y-auto">
                  {users.map((u) => {
                    const isSelected = currentUser?.user_idx === u.user_idx;
                    return (
                      <div
                        key={u.user_idx}
                        onClick={() => {
                          onSelectUser(u);
                          setProfileDropdownOpen(false);
                        }}
                        className={`flex items-center justify-between px-3 py-2 rounded-lg cursor-pointer transition ${
                          isSelected ? 'bg-surface hover:bg-surface-hover' : 'hover:bg-surface'
                        }`}
                      >
                        <div className="flex items-center space-x-2.5">
                          <div
                            className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold text-white"
                            style={{ backgroundColor: u.avatar_color }}
                          >
                            {u.display_name.charAt(0)}
                          </div>
                          <div>
                            <div className="text-xs font-medium text-white">{u.display_name}</div>
                            <div className="text-[10px] text-muted">
                              {u.preferred_genres?.slice(0, 2).join(', ') || 'Film Enthusiast'}
                            </div>
                          </div>
                        </div>
                        {isSelected && <Check className="w-4 h-4 text-accent" />}
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
                    className="w-full flex items-center justify-center space-x-2 py-2 px-3 rounded-lg bg-surface hover:bg-surface-hover border border-border text-xs text-accent font-medium transition"
                  >
                    <Plus className="w-3.5 h-3.5" />
                    <span>Create Custom Profile</span>
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};
