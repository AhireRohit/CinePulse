import { useEffect, useRef } from "react";
import type { ReactNode } from "react";
import { Check, Plus, Star, X } from "lucide-react";
import { useMovies } from "../state";
import { displayTitle } from "../moviePresentation";
export function Rating({ value }: { value: number }) {
  return (
    <span className="rating">
      <Star size={13} fill="currentColor" />
      {Number.isFinite(value) ? value.toFixed(1) : "—"}
      <span className="muted">/ 5</span>
    </span>
  );
}
export function SaveButton({
  id,
  compact = false,
}: {
  id: number;
  compact?: boolean;
}) {
  const { saved, feedback } = useMovies();
  const active = saved.includes(id);
  return (
    <button
      className={compact ? "icon-button save-button" : "button secondary"}
      aria-label={active ? "Remove from watchlist" : "Add to Watchlist"}
      aria-pressed={active}
      onClick={() => void feedback(id, "save")}
    >
      {active ? <Check size={18} /> : <Plus size={18} />}
      {!compact && (active ? "Saved to My List" : "Add to Watchlist")}
    </button>
  );
}
export function Poster({
  url,
  title,
  eager = false,
}: {
  url?: string;
  title: string;
  eager?: boolean;
}) {
  return (
    <div className="poster-image">
      <div className="poster-fallback" aria-hidden="true">
        <span>CINEPULSE COLLECTION</span>
        <strong>{displayTitle(title).split(" (")[0]}</strong>
        <span>A WORLD OF CINEMA</span>
      </div>
      {url && (
        <img
          key={url}
          src={url.replace("@..jpg", "@._V1_SX500.jpg")}
          alt={`${displayTitle(title)} poster`}
          loading={eager ? "eager" : "lazy"}
          onError={(e) => {
            e.currentTarget.style.display = "none";
          }}
        />
      )}
    </div>
  );
}
export function Dialog({
  title,
  onClose,
  children,
  className = "",
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
  className?: string;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const previous = document.activeElement as HTMLElement;
    const el = ref.current!;
    el.showModal();
    const old = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      el.close();
      document.body.style.overflow = old;
      previous?.focus();
    };
  }, []);
  return (
    <dialog
      ref={ref}
      className={`dialog ${className}`}
      aria-label={title}
      onCancel={(e) => {
        e.preventDefault();
        onClose();
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="dialog-inner">
        <button
          className="icon-button dialog-close"
          data-testid="close-modal"
          aria-label="Close dialog"
          onClick={onClose}
        >
          <X size={22} />
        </button>
        {children}
      </div>
    </dialog>
  );
}
export function EmptyState({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <div className="empty-state">
      <span className="eyebrow">THE NEXT GREAT FIND AWAITS</span>
      <h2>{title}</h2>
      <p>{children}</p>
    </div>
  );
}
export function Skeletons() {
  return (
    <div className="skeleton-grid" aria-label="Loading movies" aria-busy="true">
      {Array.from({ length: 6 }, (_, i) => (
        <div className="skeleton" key={i} />
      ))}
    </div>
  );
}
export function Pipeline() {
  return (
    <div className="pipeline">
      {[
        [
          "01",
          "Find possibilities",
          "Neural retrieval, similar audience tastes, and community favorites.",
        ],
        [
          "02",
          "Find your favorites",
          "LightGBM ranks candidates using movie and preference features.",
        ],
        [
          "03",
          "Keep it interesting",
          "A final reranking step balances relevance with genre variety.",
        ],
      ].map(([n, title, text]) => (
        <div key={n}>
          <span className="step-number">{n}</span>
          <h3>{title}</h3>
          <p>{text}</p>
        </div>
      ))}
    </div>
  );
}
