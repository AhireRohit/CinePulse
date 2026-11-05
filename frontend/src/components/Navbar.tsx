import { useState } from "react";
import {
  Aperture,
  Compass,
  Search,
  Bookmark,
  BarChart3,
  FlaskConical,
  Plus,
  ChevronDown,
  Check,
} from "lucide-react";
import type { UserProfile } from "../types";
import { Dialog } from "./Primitives";
const tabs = [
  { id: "feed", label: "Discover", icon: Aperture },
  { id: "explore", label: "Explore", icon: Compass },
  { id: "library", label: "My List", icon: Bookmark },
  { id: "insights", label: "Insights", icon: BarChart3 },
  { id: "evaluation", label: "Benchmarks", icon: FlaskConical },
];
export function Navbar({
  currentTab,
  setCurrentTab,
  users,
  currentUser,
  onSelectUser,
  onOpenOnboarding,
  onSearch,
}: {
  currentTab: string;
  setCurrentTab: (t: string) => void;
  users: UserProfile[];
  currentUser: UserProfile | null;
  onSelectUser: (u: UserProfile) => void;
  onOpenOnboarding: () => void;
  onSearch: () => void;
}) {
  const [profiles, setProfiles] = useState(false);
  return (
    <>
      <header className="site-header">
        <div className="nav-inner">
          <button
            className="wordmark"
            aria-label="CinePulse home"
            onClick={() => setCurrentTab("feed")}
          >
            <span className="brand-symbol">
              <Aperture size={23} />
            </span>
            cinepulse<span className="brand-period">.</span>
          </button>
          <nav className="desktop-nav" aria-label="Main navigation">
            {tabs.map((t) => (
              <button
                key={t.id}
                aria-current={currentTab === t.id ? "page" : undefined}
                className={`${currentTab === t.id ? "active" : ""} ${["insights", "evaluation"].includes(t.id) ? "technical-link" : ""}`}
                onClick={() => setCurrentTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </nav>
          <div className="nav-actions">
            <button
              className="search-trigger"
              aria-label="Search movies"
              onClick={onSearch}
            >
              <Search size={19} />
              <span>Search films</span>
              <kbd>⌘ K</kbd>
            </button>
            <span className="nav-divider" />
            <button
              className="profile-trigger"
              aria-label={`Switch profile: ${currentUser?.display_name || "Select profile"}`}
              onClick={() => setProfiles(true)}
            >
              <span
                className="avatar"
                style={{ background: currentUser?.avatar_color }}
              >
                {currentUser?.display_name.charAt(0) || "?"}
              </span>
              <span className="profile-name">
                {currentUser?.display_name.split(" ")[0]}
              </span>
              <ChevronDown size={14} />
            </button>
          </div>
        </div>
      </header>
      <nav className="mobile-nav" aria-label="Mobile navigation">
        {tabs.map((t) => (
          <button
            key={t.id}
            aria-current={currentTab === t.id ? "page" : undefined}
            className={currentTab === t.id ? "active" : ""}
            onClick={() => setCurrentTab(t.id)}
          >
            <t.icon size={20} />
            <span>{t.label}</span>
          </button>
        ))}
      </nav>
      {profiles && (
        <Dialog
          title="Choose your profile"
          onClose={() => setProfiles(false)}
          className="profile-dialog"
        >
          <div className="dialog-padding">
            <span className="eyebrow">A DIFFERENT POINT OF VIEW</span>
            <h2>Who’s discovering?</h2>
            <p className="muted">Every taste tells a different story.</p>
            <div className="profile-list">
              {users.map((u) => (
                <button
                  key={u.user_idx}
                  className="profile-option"
                  onClick={() => {
                    onSelectUser(u);
                    setProfiles(false);
                  }}
                >
                  <span
                    className="avatar"
                    style={{ background: u.avatar_color }}
                  >
                    {u.display_name[0]}
                  </span>
                  <span>
                    <strong>{u.display_name}</strong>
                    <small>{u.preferred_genres.join(" · ")}</small>
                  </span>
                  {u.user_idx === currentUser?.user_idx && (
                    <Check size={18} className="accent" />
                  )}
                </button>
              ))}
            </div>
            <button
              className="button secondary full-width"
              onClick={() => {
                setProfiles(false);
                onOpenOnboarding();
              }}
            >
              <Plus size={18} /> Create a profile
            </button>
          </div>
        </Dialog>
      )}
    </>
  );
}
