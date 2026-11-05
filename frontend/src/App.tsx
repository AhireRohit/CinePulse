import { useCallback, useEffect, useRef, useState } from "react";
import type { UserProfile, RecommendationFeed } from "./types";
import {
  getUsers,
  getRecommendations,
  getUserSaved,
  postFeedback,
  isStaticHost,
} from "./api/client";
import { MovieState, readStored } from "./state";
import type { FeedbackAction } from "./state";
import { Navbar } from "./components/Navbar";
import { HomePage } from "./pages/HomePage";
import { ExplorePage } from "./pages/ExplorePage";
import { LibraryPage } from "./pages/LibraryPage";
import { InsightsPage } from "./pages/InsightsPage";
import { EvaluationPage } from "./pages/EvaluationPage";
import { MovieDetailModal } from "./components/MovieDetailModal";
import { OnboardingModal } from "./components/OnboardingModal";
import { GlobalSearchBar } from "./components/GlobalSearchBar";
export default function App() {
  const [tab, setTab] = useState("feed");
  const [users, setUsers] = useState<UserProfile[]>([]);
  const [user, setUser] = useState<UserProfile | null>(null);
  const [feed, setFeed] = useState<RecommendationFeed | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [movieId, setMovieId] = useState<number | null>(null);
  const [onboarding, setOnboarding] = useState(false);
  const [search, setSearch] = useState(false);
  const [notice, setNotice] = useState("");
  const [saved, setSaved] = useState<number[]>([]);
  const [liked, setLiked] = useState<number[]>([]);
  const [disliked, setDisliked] = useState<number[]>([]);
  const [ratings, setRatings] = useState<Record<number, number>>({});
  const request = useRef(0);
  const activeUser = useRef<number | undefined>(undefined);
  const navigate = (next: string) => {
    setTab(next);
    window.scrollTo({ top: 0 });
  };
  useEffect(() => {
    getUsers()
      .then((data) => {
        setUsers(data);
        setUser(
          data.find(
            (u) => u.user_idx === readStored("cinepulse_profile", -1),
          ) ||
            data[0] ||
            null,
        );
        if (!data.length)
          setError("Profiles are unavailable. Please reload to try again.");
      })
      .catch(() =>
        setError("Profiles couldn’t load. Please reload to try again."),
      );
  }, []);
  const loadFeed = useCallback(
    (diversity = 0.85, novelty = 0.03) => {
      if (!user) return;
      const seq = ++request.current;
      setLoading(true);
      getRecommendations(user.user_idx, { diversity, novelty })
        .then((data) => {
          if (seq === request.current) {
            setFeed(data);
            setError("");
          }
        })
        .catch(() => {
          if (seq === request.current) {
            setFeed(null);
            setError("Your recommendations couldn’t load.");
          }
        })
        .finally(() => {
          if (seq === request.current) setLoading(false);
        });
    },
    [user],
  );
  useEffect(() => {
    activeUser.current = user?.user_idx;
    if (!user) return;
    setFeed(null);
    setMovieId(null);
    setNotice("");
    loadFeed();
    const id = user.user_idx;
    localStorage.setItem("cinepulse_profile", JSON.stringify(id));
    const savedLocal = readStored<number[]>(`cinepulse_saved_${id}`, []);
    setSaved(savedLocal);
    setLiked(readStored(`cinepulse_liked_${id}`, []));
    setDisliked(readStored(`cinepulse_disliked_${id}`, []));
    setRatings(readStored(`cinepulse_rated_${id}`, {}));
    if (!isStaticHost)
      getUserSaved(id)
        .then((items) => {
          if (activeUser.current !== id) return;
          const removed = readStored<number[]>(`cinepulse_removed_${id}`, []);
          const combined = [
            ...new Set([
              ...savedLocal,
              ...items.map((m: { movie_idx: number }) => m.movie_idx),
            ]),
          ].filter((x) => !removed.includes(x));
          setSaved(combined);
          localStorage.setItem(
            `cinepulse_saved_${id}`,
            JSON.stringify(combined),
          );
        })
        .catch(() => setNotice("Your backend watchlist could not be loaded."));
    return () => {
      ++request.current;
    };
  }, [user, loadFeed]);
  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(""), 3200);
    return () => clearTimeout(timer);
  }, [notice]);
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        if (movieId === null && !onboarding) setSearch((s) => !s);
      }
    };
    document.addEventListener("keydown", key);
    return () => document.removeEventListener("keydown", key);
  }, [movieId, onboarding]);
  const feedback = async (
    id: number,
    action: FeedbackAction,
    rating?: number,
  ) => {
    if (!user) return;
    const uid = user.user_idx;
    const removing = action === "save" && saved.includes(id);
    try {
      if (!removing) await postFeedback(uid, id, action, rating);
      const key = `cinepulse_${action === "save" ? "saved" : action === "like" ? "liked" : action === "dislike" ? "disliked" : action === "rate" ? "rated" : "dismissed"}_${uid}`;
      if (action === "rate") {
        const next = {
          ...readStored<Record<number, number>>(key, {}),
          [id]: rating!,
        };
        localStorage.setItem(key, JSON.stringify(next));
        if (activeUser.current === uid) setRatings(next);
      } else {
        const prev = readStored<number[]>(key, []);
        const next = removing
          ? prev.filter((x) => x !== id)
          : [...new Set([...prev, id])];
        localStorage.setItem(key, JSON.stringify(next));
        if (action === "save") {
          const removedKey = `cinepulse_removed_${uid}`;
          const prevRemoved = readStored<number[]>(removedKey, []);
          localStorage.setItem(
            removedKey,
            JSON.stringify(
              removing
                ? [...new Set([...prevRemoved, id])]
                : prevRemoved.filter((x) => x !== id),
            ),
          );
        }
        if (activeUser.current === uid) {
          if (action === "save") setSaved(next);
          if (action === "like") setLiked(next);
          if (action === "dislike") setDisliked(next);
        }
      }
      if (activeUser.current !== uid) return;
      setNotice(
        removing
          ? "Removed from your list."
          : action === "save"
            ? "Saved for another evening."
            : action === "rate"
              ? `Your ${rating}-star rating is saved.`
              : action === "like"
                ? "Added to your liked films."
                : "Hidden from your discoveries.",
      );
      if (action === "dislike" || action === "dismiss")
        setFeed((f) =>
          f
            ? {
                ...f,
                hero_featured:
                  f.hero_featured?.movie_idx === id
                    ? undefined
                    : f.hero_featured,
                rails: Object.fromEntries(
                  Object.entries(f.rails).map(([name, items]) => [
                    name,
                    items.filter((m) => m.movie_idx !== id),
                  ]),
                ),
              }
            : f,
        );
      if (!isStaticHost && action !== "save") loadFeed();
    } catch {
      setNotice("Couldn’t save that change. Please try again.");
    }
  };
  const recommendation = feed
    ? [feed.hero_featured, ...Object.values(feed.rails).flat()].find(
        (m) => m?.movie_idx === movieId,
      )
    : undefined;
  return (
    <MovieState.Provider value={{ saved, liked, disliked, ratings, feedback }}>
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <Navbar
        currentTab={tab}
        setCurrentTab={navigate}
        users={users}
        currentUser={user}
        onSelectUser={setUser}
        onOpenOnboarding={() => setOnboarding(true)}
        onSearch={() => setSearch(true)}
      />
      <main id="main">
        {error && (
          <div className="error-banner" role="alert">
            {error} <button onClick={() => loadFeed()}>Retry</button>
          </div>
        )}
        {tab === "feed" && (
          <HomePage
            feed={feed}
            loading={loading}
            onRefresh={loadFeed}
            onOpenDetail={setMovieId}
            activeProfileName={user?.display_name || "you"}
          />
        )}
        {tab === "explore" && <ExplorePage onOpenDetail={setMovieId} />}
        {tab === "library" && (
          <LibraryPage
            currentUser={user}
            onOpenDetail={setMovieId}
            onExplore={() => navigate("explore")}
          />
        )}
        {tab === "insights" && <InsightsPage currentUser={user} />}
        {tab === "evaluation" && <EvaluationPage />}
      </main>
      <footer className="content-width site-footer">
        <div>
          <span className="footer-brand">
            cinepulse<span>.</span>
          </span>
          <p>For the love of finding a great film.</p>
        </div>
      </footer>
      {notice && (
        <div className="toast" role="status">
          {notice}
        </div>
      )}
      {search && (
        <GlobalSearchBar
          onClose={() => setSearch(false)}
          onOpenDetail={setMovieId}
        />
      )}
      <MovieDetailModal
        key={movieId ?? "closed"}
        movieId={movieId}
        onClose={() => setMovieId(null)}
        recommendation={recommendation}
        onOpenDetail={setMovieId}
      />
      {onboarding && (
        <OnboardingModal
          onClose={() => setOnboarding(false)}
          onUserCreated={(u) => {
            setUsers((prev) => [...prev, u]);
            setUser(u);
            navigate("feed");
          }}
        />
      )}
    </MovieState.Provider>
  );
}
