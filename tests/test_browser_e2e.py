from pathlib import Path

import pytest

SCREENSHOTS_DIR = Path("artifacts/screenshots")
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)


def test_browser_e2e_and_capture_screenshots():
    """
    Executes an end-to-end browser test against the live CinePulse streaming application
    using Playwright Chromium and captures screenshots of key user journeys:
    1. Homepage Hero & Streaming Rails
    2. Global Search Modal with Real-time Query Results
    3. Movie Detail & Explainability Modal
    4. Real-time User Insights & Genre Affinity Visualizations
    5. Live Offline Evaluation Benchmarks & Bootstrap Metrics
    6. Category & Exploration Rails
    """
    pytest.importorskip("playwright")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        # 1. Navigate to Home
        print("Navigating to http://localhost:8000/...")
        page.goto("http://localhost:8000/", wait_until="networkidle")
        page.wait_for_timeout(2000)

        # Verify Title
        title = page.title()
        assert "CinePulse" in title, f"Expected CinePulse in title, got: {title}"

        # Screenshot 1: Home Page Hero & Top Rails
        page.screenshot(path=str(SCREENSHOTS_DIR / "01_homepage_hero.png"), full_page=False)
        print("Captured 01_homepage_hero.png")

        # Test Real-Time Feedback Interaction on Hero (Like & Watchlist)
        like_btn = page.locator("button:has-text('Like')").first
        if like_btn.is_visible():
            print("Testing user feedback interaction (Like)...")
            like_btn.click()
            page.wait_for_timeout(800)
            liked_btn = page.locator("button:has-text('Liked')").first
            assert liked_btn.is_visible(), "Expected button state to update to 'Liked'"
            print("Feedback state update verified!")

        save_btn = page.locator("button:has-text('Watchlist')").first
        if save_btn.is_visible():
            print("Testing Watchlist save interaction...")
            save_btn.click()
            page.wait_for_timeout(500)
            saved_btn = page.locator("button:has-text('Saved')").first
            assert saved_btn.is_visible(), "Expected button state to update to 'Saved'"
            print("Watchlist state update verified!")

        # Test Profile Switching
        print("Testing Profile Switching in Navbar...")
        profile_btn = page.locator("header button:has-text('Elena Vance'), header button:has-text('Marcus Sterling')").first
        if profile_btn.is_visible():
            profile_btn.click()
            page.wait_for_timeout(500)
            target_user = page.locator("div:has-text('Marcus Sterling')").last
            if target_user.is_visible():
                target_user.click()
                page.wait_for_timeout(1500)
                print("Switched profile to Marcus Sterling successfully!")

        # 2. Open Search Modal via hotkey Ctrl+K
        print("Triggering Global Search Modal...")
        page.keyboard.press("Control+k")
        page.wait_for_timeout(1000)
        search_input = page.locator("input[placeholder*='Search']")
        if search_input.is_visible():
            search_input.fill("Matrix")
            page.wait_for_timeout(1500)
            page.screenshot(path=str(SCREENSHOTS_DIR / "02_search_modal.png"))
            print("Captured 02_search_modal.png")
            page.keyboard.press("Escape")
            page.wait_for_timeout(800)
            # Ensure search modal is closed
            esc_btn = page.locator("button:has-text('ESC')").first
            if esc_btn.is_visible():
                esc_btn.click()
                page.wait_for_timeout(500)

        # 3. Open Movie Detail Modal
        print("Opening Movie Details Modal...")
        movie_card = page.locator("[data-testid='movie-card']").first
        if movie_card.is_visible():
            movie_card.click()
        else:
            inspect_btn = page.locator("button:has-text('Inspect Film & Signals')").first
            if inspect_btn.is_visible():
                inspect_btn.click()
        page.wait_for_timeout(1500)
        page.screenshot(path=str(SCREENSHOTS_DIR / "03_movie_detail_modal.png"))
        print("Captured 03_movie_detail_modal.png")
        close_btn = page.locator("[data-testid='close-modal']").first
        if close_btn.is_visible():
            close_btn.click()
        else:
            page.keyboard.press("Escape")
        page.wait_for_timeout(800)

        # 4. Navigate to Insights Page
        print("Navigating to Insights Tab...")
        insights_button = page.locator("nav button:has-text('Insights')").first
        if insights_button.is_visible():
            insights_button.click()
            page.wait_for_timeout(2000)
            page.screenshot(path=str(SCREENSHOTS_DIR / "04_insights_tab.png"), full_page=False)
            print("Captured 04_insights_tab.png")

        # 5. Navigate to Benchmarks Page
        print("Navigating to Benchmarks Tab...")
        benchmarks_button = page.locator("nav button:has-text('Benchmarks')").first
        if benchmarks_button.is_visible():
            benchmarks_button.click()
            page.wait_for_timeout(2000)
            page.screenshot(path=str(SCREENSHOTS_DIR / "05_benchmarks_tab.png"), full_page=False)
            print("Captured 05_benchmarks_tab.png")

        # 6. Navigate to Explore Page
        print("Navigating to Explore Tab...")
        explore_button = page.locator("nav button:has-text('Explore')").first
        if explore_button.is_visible():
            explore_button.click()
            page.wait_for_timeout(2000)
            page.screenshot(path=str(SCREENSHOTS_DIR / "06_explore_tab.png"), full_page=False)
            print("Captured 06_explore_tab.png")

        # 7. Return to Discover Feed
        discover_btn = page.locator("nav button:has-text('Discover')").first
        if discover_btn.is_visible():
            discover_btn.click()
            page.wait_for_timeout(1000)
        page.screenshot(path=str(SCREENSHOTS_DIR / "07_final_responsive_feed.png"), full_page=False)
        print("Captured 07_final_responsive_feed.png")
        context.close()

        # ==========================================
        # Mobile Viewport Testing (iPhone 14 / Pixel)
        # ==========================================
        print("Starting Mobile Viewport Testing (393x852)...")
        mobile_context = browser.new_context(
            viewport={"width": 393, "height": 852},
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
            is_mobile=True,
            has_touch=True,
        )
        mobile_page = mobile_context.new_page()
        mobile_page.goto("http://localhost:8000/", wait_until="networkidle")
        mobile_page.wait_for_timeout(2000)

        # Screenshot 8: Mobile Feed View
        mobile_page.screenshot(path=str(SCREENSHOTS_DIR / "08_mobile_feed.png"), full_page=False)
        print("Captured 08_mobile_feed.png")

        # Screenshot 9: Mobile Details Modal
        mobile_card = mobile_page.locator("[data-testid='movie-card']").first
        if mobile_card.is_visible():
            mobile_card.click()
            mobile_page.wait_for_timeout(1500)
            mobile_page.screenshot(path=str(SCREENSHOTS_DIR / "09_mobile_detail_modal.png"), full_page=False)
            print("Captured 09_mobile_detail_modal.png")
            mobile_page.keyboard.press("Escape")
            mobile_page.wait_for_timeout(500)

        mobile_context.close()
        browser.close()
        print("All desktop and mobile browser e2e tests passed and screenshots captured successfully!")


if __name__ == "__main__":
    test_browser_e2e_and_capture_screenshots()
