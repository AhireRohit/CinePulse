"""Verify the published consumer UI using Chromium, including mobile interactions."""

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "https://ahirerohit.github.io/CinePulse/"
OUTPUT = Path("artifacts/screenshots/public")
OUTPUT.mkdir(parents=True, exist_ok=True)
FORBIDDEN = [
    "Precomputed V2 demo",
    "Demo profiles show exported V2 recommendations.",
    "Built with curiosity. Powered by MovieLens.",
]


def check(page):
    text = page.locator("body").inner_text()
    assert all(phrase not in text for phrase in FORBIDDEN)
    assert page.locator(".discovery-intro, .demo-note").count() == 0
    assert page.locator('footer a[href*="github.com"]').count() == 0
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")


def nav(page, name):
    page.locator("nav:visible").get_by_role("button", name=name, exact=True).click()


def main():
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for label, width, height in [("desktop", 1440, 900), ("mobile", 390, 844)]:
            context = browser.new_context(viewport={"width": width, "height": height})
            page = context.new_page()
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(URL, wait_until="networkidle")
            page.locator(".hero h1").wait_for()
            check(page)
            page.screenshot(path=str(OUTPUT / f"{label}-discover.png"))
            page.get_by_role("button", name="View Details", exact=True).click()
            page.locator(".detail-copy h2").wait_for()
            page.get_by_role("button", name="Rate 4 stars", exact=True).click()
            page.screenshot(path=str(OUTPUT / f"{label}-details.png"))
            page.keyboard.press("Escape")
            page.locator(".hero-actions").get_by_role("button", name="Add to Watchlist").click()
            nav(page, "My List")
            page.locator(".movie-card").wait_for()
            check(page)
            page.get_by_role("button", name="Remove from watchlist").click()
            page.get_by_role("heading", name="Make room for great films.").wait_for()
            page.get_by_role("button", name="Search movies", exact=True).click()
            page.get_by_placeholder("Search by title, genre, or year", exact=True).fill("Matrix")
            page.locator(".search-result").first.wait_for()
            assert "Matrix" in page.locator(".search-result").first.inner_text()
            page.keyboard.press("Escape")
            page.get_by_role("button", name="Switch profile:", exact=False).click()
            page.locator(".profile-option").filter(has_text="Marcus Sterling").click()
            nav(page, "Discover")
            page.locator(".hero h1").wait_for()
            assert "Marcus" in page.locator(".collection-intro").inner_text()
            nav(page, "Explore")
            page.get_by_text("3,416", exact=True).wait_for()
            check(page)
            nav(page, "Insights")
            page.locator(".genre-bars").wait_for()
            check(page)
            nav(page, "Benchmarks")
            assert "27.92" in page.locator("main").inner_text()
            assert page.locator("tbody tr").count() == 3
            check(page)
            page.screenshot(path=str(OUTPUT / f"{label}-benchmarks.png"))
            context.close()
            print(f"{label}: homepage, details, ratings, watchlist, search, profiles, catalog, insights, benchmarks PASS")
        browser.close()
    assert not errors, errors
    (OUTPUT / "verification.json").write_text(json.dumps({"url": URL, "passed": True, "browser_errors": errors}, indent=2))
    print(f"Public UI verification passed: {URL}")


if __name__ == "__main__":
    main()
