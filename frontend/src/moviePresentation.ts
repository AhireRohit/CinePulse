export function displayTitle(title: string) {
  const m = title.match(/^(.*), (The|A|An)$/);
  return m ? `${m[2]} ${m[1]}` : title;
}
export const artwork = (movie: { movieId: number; backdrop_url?: string }) =>
  movie.backdrop_url ||
  (movie.movieId === 858
    ? `${import.meta.env.BASE_URL}artwork/godfather-backdrop.jpg`
    : undefined);
