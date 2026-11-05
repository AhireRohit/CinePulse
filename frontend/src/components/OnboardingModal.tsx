import { useState } from "react";
import { ArrowRight, Check } from "lucide-react";
import type { UserProfile } from "../types";
import { createUser, isStaticHost } from "../api/client";
import { Dialog } from "./Primitives";
const genres = [
  "Sci-Fi",
  "Mystery",
  "Crime",
  "Film-Noir",
  "Drama",
  "Thriller",
  "Action",
  "Adventure",
  "Animation",
  "Comedy",
  "Fantasy",
  "Romance",
  "Horror",
  "Documentary",
  "War",
  "Western",
];
export function OnboardingModal({
  onClose,
  onUserCreated,
}: {
  onClose: () => void;
  onUserCreated: (u: UserProfile) => void;
}) {
  const [name, setName] = useState("");
  const [selected, setSelected] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  return (
    <Dialog
      title="Create your profile"
      onClose={onClose}
      className="onboarding-dialog"
    >
      <form
        className="dialog-padding"
        onSubmit={async (e) => {
          e.preventDefault();
          if (!name.trim()) {
            setError("Enter your name to create a profile.");
            return;
          }
          if (!selected.length) {
            setError("Choose at least one genre to get started.");
            return;
          }
          setBusy(true);
          setError("");
          try {
            const user = await createUser({
              username:
                name.toLowerCase().replace(/[^a-z0-9]/g, "_") ||
                `film_fan_${Date.now()}`,
              display_name: name.trim(),
              preferred_genres: selected,
              favorite_movies: [],
              avatar_color: "#c86148",
            });
            onUserCreated(user);
            onClose();
          } catch {
            setError("Couldn’t create your profile. Please try again.");
          } finally {
            setBusy(false);
          }
        }}
      >
        <span className="eyebrow">YOUR TASTE. YOUR NEXT CHAPTER.</span>
        <h2>Let’s find your kind of film.</h2>
        <p className="muted">
          A few things you love. A whole world to discover.
        </p>
        <label className="form-label" htmlFor="profile-name">
          What should we call you?
        </label>
        <input
          id="profile-name"
          autoFocus
          className="text-input"
          placeholder="Your name"
          required
          maxLength={60}
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <fieldset>
          <legend>
            What draws you in?{" "}
            <span className="muted">Choose your genres.</span>
          </legend>
          <div className="onboarding-genres">
            {genres.map((g) => (
              <button
                type="button"
                key={g}
                aria-pressed={selected.includes(g)}
                className={selected.includes(g) ? "selected" : ""}
                onClick={() =>
                  setSelected((s) =>
                    s.includes(g) ? s.filter((x) => x !== g) : [...s, g],
                  )
                }
              >
                {g}
                {selected.includes(g) && <Check size={15} />}
              </button>
            ))}
          </div>
        </fieldset>
        <p className="onboarding-note">
          {isStaticHost
            ? "Your profile is saved in this browser. We’ll curate films by your chosen genres and community ratings. Neural inference is available with the live backend."
            : "Your preferences give the connected recommendation engine a starting point."}
        </p>
        {error && (
          <p role="alert" className="accent">
            {error}
          </p>
        )}
        <button
          type="submit"
          className="button primary full-width"
          disabled={busy}
        >
          {busy ? "Creating your profile…" : "Start discovering"}
          <ArrowRight size={18} />
        </button>
      </form>
    </Dialog>
  );
}
