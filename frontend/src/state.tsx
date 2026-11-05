import { createContext, useContext } from "react";
export type FeedbackAction = "like" | "dislike" | "save" | "dismiss" | "rate";
export const MovieState = createContext<{
  saved: number[];
  liked: number[];
  disliked: number[];
  ratings: Record<number, number>;
  feedback: (
    id: number,
    action: FeedbackAction,
    rating?: number,
  ) => Promise<void>;
}>({
  saved: [],
  liked: [],
  disliked: [],
  ratings: {},
  feedback: async () => {},
});
export const useMovies = () => useContext(MovieState);
export function readStored<T>(key: string, fallback: T): T {
  try {
    return JSON.parse(localStorage.getItem(key) || "null") ?? fallback;
  } catch {
    return fallback;
  }
}
