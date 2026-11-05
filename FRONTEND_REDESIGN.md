# CinePulse frontend redesign

Implemented on 5 November 2026. Final UI cleanup removes the demo information strip and footer attribution before release to the existing GitHub Pages site.

## Design and behavior

The interface now centers film discovery: charcoal surfaces, warm typography, a coral accent, an editorial serif spotlight, consistent 2:3 posters, quiet metadata, and a dedicated mobile bottom navigation. Shared components cover dialogs, posters, ratings, saving, rails, loading, empty states, and the recommendation pipeline.

- Discover retains the actual exported recommendation rails and their item order. The first rail is presented as “Recommended for you.” The existing exports contain four rails; a fifth recommendation source was not invented.
- Explore opens the complete 3,416-film catalog with an empty query, rather than searching for “the.” Title, genre, and year search works with genre and era filters, popularity/rating/year sorting, grid/list views, counts, pagination, and empty states.
- My List uses one profile-specific state shared by the hero, cards, details, and library. Saves, removal, likes, dislikes, numerical ratings, custom profiles, and the selected profile survive reloads in this browser.
- Details use genuine catalog metadata. Unavailable synopsis/cast/director/streaming information is omitted. Static genre overlap is clearly labeled “More in these genres,” rather than collaborative-filter similarity. The live backend's genuine similarity response remains available.
- Native modal dialogs provide focus containment, Escape dismissal, scroll locking, backdrop dismissal, and focus restoration, including under React StrictMode. Keyboard focus indicators and reduced-motion styling are included.
- Insights separates exported candidate counts and recorded timings from local profile preferences. Equal genre bars are explicitly labeled declared preferences, rather than fabricated learned affinity. New profiles are accurately described as local genre curation.
- Benchmarks tells the V1/V2 story with comparative charts, architecture, six experiment families, diversity/accuracy/latency tradeoffs, and an expandable full table. Its data is copied from the original 3,974-user confirmation artifact without modifying historical files. Headline values are 0.0560 → 0.0778 NDCG@10, 18.09% → 22.19% Recall@20, 95.08% V2 Coverage@100, and 27.92 ms V2 P95. Diversity falls from 0.6933 to 0.5165 and is explained explicitly. Validation experiments are identified separately from the test cohort.
- Static mode hides inference tuning. The homepage demo strip has been removed; contextual explanations in onboarding, Insights, and movie details still accurately describe local curation and recorded results. The live backend retains recommendation tuning and feedback. Failed live mutations are reported instead of silently pretending to succeed locally.
- ID matching consistently uses `movie_idx`; a coincidental `movieId` match no longer saves, hides, or opens the wrong film.

## Actual screenshots

All files are Playwright screenshots of the implemented application, not design mockups.

- [Before / after](artifacts/screenshots/final/before-after.jpg)
- [Desktop overview](artifacts/screenshots/final/contact-desktop.jpg)
- [Mobile overview](artifacts/screenshots/final/contact-mobile.jpg)
- [Tablet overview](artifacts/screenshots/final/contact-tablet.jpg)
- [Wide desktop overview](artifacts/screenshots/final/contact-wide.jpg)

| Screen | Desktop, 1440 × 900 | Mobile, 390 × 844 |
| --- | --- | --- |
| Discover | [Screenshot](artifacts/screenshots/final/desktop-discover.png) | [Screenshot](artifacts/screenshots/final/mobile-discover.png) |
| Explore | [Screenshot](artifacts/screenshots/final/desktop-explore.png) | [Screenshot](artifacts/screenshots/final/mobile-explore.png) |
| Movie details | [Screenshot](artifacts/screenshots/final/desktop-details.png) | [Screenshot](artifacts/screenshots/final/mobile-details.png) |
| My List | [Screenshot](artifacts/screenshots/final/desktop-my-list.png) | [Screenshot](artifacts/screenshots/final/mobile-my-list.png) |
| Insights | [Screenshot](artifacts/screenshots/final/desktop-insights.png) | [Screenshot](artifacts/screenshots/final/mobile-insights.png) |
| Benchmarks | [Screenshot](artifacts/screenshots/final/desktop-benchmarks.png) | [Screenshot](artifacts/screenshots/final/mobile-benchmarks.png) |
| Profiles | [Screenshot](artifacts/screenshots/final/desktop-profiles.png) | [Screenshot](artifacts/screenshots/final/mobile-profiles.png) |
| Onboarding | [Screenshot](artifacts/screenshots/final/desktop-onboarding.png) | [Screenshot](artifacts/screenshots/final/mobile-onboarding.png) |

The final directory also contains individual tablet (768 × 1024) and wide desktop (1920 × 1080) captures, full-page desktop/mobile captures, and live-backend smoke-test captures. The `before`, `pass1`, and `pass2` directories preserve the visual review stages.

## Visual review and responsive verification

Initial implementation was followed by two refinement passes and a final capture:

1. Reduced hero and introductory spacing so posters enter the first desktop screen; restored native dialog centering after the CSS reset; enlarged rail and save targets.
2. Compacted mobile onboarding until its submit button fit in the tested viewport; increased close/rating targets; corrected StrictMode focus restoration; improved typographic poster fallbacks; let hero height accommodate longer featured titles.

Reviewed all eight screens at all four requested viewports. Automated capture checks assert no horizontal document or dialog overflow. The four preset profiles were also switched at desktop and mobile widths to inspect longer hero titles. Narrow layouts use two-column poster grids and scrollable rails; tablet uses three columns; wider layouts use five or six columns. Filter controls wrap, and benchmark tables scroll within their own container.

## Validation

- `npm run build`: PASS, including TypeScript project checking and production Vite build.
- `npx tsc -b --pretty false`: PASS.
- `npm run lint`: exits successfully; advisory React effect warnings remain, including the untouched legacy SessionDrawer. No lint errors.
- Existing browser E2E, `tests/test_browser_e2e.py`: PASS.
- New interaction regressions, `tests/test_frontend_redesign.py`: 5 PASS. Covers saving/removal across views, profiles, reloads, combined filters, list view, empty state, hotkey search, modal focus containment/restoration, persisted ratings, dislike suppression, ID namespace collisions, onboarding, static controls, and cohort values.
- Existing backend lifecycle E2E, `tests/test_e2e_journey.py`: PASS.
- Backend regression selection (`test_api.py`, `test_product_safeguards.py`, `test_serving_parity.py`, `test_session_adaptation.py`): 23 PASS.
- Combined total: 30 passing tests across the above runs. Existing Python dependency/deprecation warnings remain.
- Browser JavaScript errors during four-viewport capture: none (`browser-errors.json`).
- Live FastAPI UI smoke check: live tuning visible, static label absent, recommendation refresh returns 200, rating feedback returns 200.
- Production build served under `/CinePulse/`: relative assets, featured artwork, static fallback, complete catalog, and navigation verified with Playwright.
- Backend model code and historical evaluation artifacts were not modified. No dependencies were added to package.json or the lockfile.

## Changed files

Modified:

- `frontend/index.html`
- `frontend/src/App.tsx`
- `frontend/src/api/client.ts`
- `frontend/src/index.css`
- `frontend/src/components/GlobalSearchBar.tsx`
- `frontend/src/components/HeroBanner.tsx`
- `frontend/src/components/MovieCard.tsx`
- `frontend/src/components/MovieDetailModal.tsx`
- `frontend/src/components/MovieRail.tsx`
- `frontend/src/components/Navbar.tsx`
- `frontend/src/components/OnboardingModal.tsx`
- `frontend/src/pages/HomePage.tsx`
- `frontend/src/pages/ExplorePage.tsx`
- `frontend/src/pages/LibraryPage.tsx`
- `frontend/src/pages/InsightsPage.tsx`
- `frontend/src/pages/EvaluationPage.tsx`
- Existing E2E screenshot outputs under `artifacts/screenshots/01_…09_…` were refreshed by the existing test.

Added:

- `frontend/src/components/Primitives.tsx`
- `frontend/src/moviePresentation.ts`
- `frontend/src/state.tsx`
- `frontend/src/data/benchmark.json`
- `frontend/public/artwork/godfather-backdrop.jpg`
- `frontend/public/artwork/README.md`
- `tests/capture_frontend.py`
- `tests/test_frontend_redesign.py`
- This report and screenshots under `artifacts/screenshots/{before,pass1,pass2,final}`.

## Limitations

- Some existing external poster URLs are unavailable; the poster fallback is deliberate and consistent. No additional metadata service was added. One verified film-specific backdrop is bundled locally; other featured films use their existing posters when a backdrop is unavailable.
- The GitHub Pages demo does not run Python inference. Likes and ratings are persisted locally; dislikes suppress exported results. Custom profiles use genre/community-rating curation. Browser storage does not synchronize across devices.
- The existing backend has no unsave endpoint. Removing a saved film uses a persistent browser-local removal override, including when the backend is active; it does not delete a remote saved row. Adding still uses the genuine backend save endpoint.
- Explore uses the bundled V2 catalog by default. With an active backend, nonempty queries use its existing search endpoint, so backend search result limits remain in effect.
- The genre chart uses declared preferences because the static insight exports do not contain learned genre affinity values. Export timing details remain clearly labeled recorded timings; no live timing is fabricated.
- Automated visual review used Chromium. Native dialog behavior was exercised through Chromium Playwright; separate Safari/Firefox testing was not performed.

## Run locally (PowerShell)

From the repository root, run the static demo:

```powershell
cd frontend
npm ci
$env:VITE_DEMO_MODE = 'true'
npm run dev -- --host localhost --port 8000
```

Open http://localhost:8000/.

To build and run against the existing FastAPI backend:

```powershell
cd frontend
Remove-Item Env:VITE_DEMO_MODE -ErrorAction SilentlyContinue
npm run build
cd ..
.venv\Scripts\python.exe -m uvicorn src.api.main:app --host 127.0.0.1 --port 8001
```

Open http://127.0.0.1:8001/. The backend serves the built frontend and its same-origin API. Existing Python dependencies and model/data artifacts are required.

With the static demo running on port 8000, reproduce review captures and browser tests from the repository root:

```powershell
.venv\Scripts\python.exe tests/capture_frontend.py final
.venv\Scripts\python.exe -m pytest tests/test_frontend_redesign.py tests/test_browser_e2e.py -q
```

## Final release cleanup

Removed the entire homepage demo information strip, including its label and explanation, plus the footer attribution. The footer contains no GitHub repository link. Removed unused styling and updated the regression test to require these elements to be absent. The cinematic layout and functional static behavior are preserved.
