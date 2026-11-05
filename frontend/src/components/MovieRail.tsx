import { useRef } from "react";
import { ArrowLeft, ArrowRight } from "lucide-react";
import type { RecommendationItem } from "../types";
import { MovieCard } from "./MovieCard";
interface Props {
  title: string;
  subtitle?: string;
  movies: RecommendationItem[];
  onOpenDetail: (id: number) => void;
}
export function MovieRail({ title, subtitle, movies, onOpenDetail }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  if (!movies.length) return null;
  return (
    <section className="movie-section">
      <div className="section-heading">
        <div>
          <h2>{title}</h2>
          {subtitle && <p>{subtitle}</p>}
        </div>
        <div className="rail-controls">
          <button
            className="icon-button"
            aria-label={`Scroll ${title} left`}
            onClick={() =>
              ref.current?.scrollBy({ left: -600, behavior: "smooth" })
            }
          >
            <ArrowLeft size={18} />
          </button>
          <button
            className="icon-button"
            aria-label={`Scroll ${title} right`}
            onClick={() =>
              ref.current?.scrollBy({ left: 600, behavior: "smooth" })
            }
          >
            <ArrowRight size={18} />
          </button>
        </div>
      </div>
      <div ref={ref} className="movie-rail">
        {movies.map((m) => (
          <MovieCard key={m.movie_idx} movie={m} onOpenDetail={onOpenDetail} />
        ))}
      </div>
    </section>
  );
}
