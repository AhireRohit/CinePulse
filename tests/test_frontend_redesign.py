"""Behavioral regressions for the locally running frontend (port 8000)."""
import json

import pytest


@pytest.fixture()
def page():
    pytest.importorskip('playwright')
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport={'width': 1440, 'height': 900})
        page = context.new_page()
        page.goto('http://localhost:8000', wait_until='networkidle')
        page.locator('.hero h1').wait_for()
        yield page
        context.close()
        browser.close()


def nav(page, label):
    page.locator('nav:visible').get_by_role('button', name=label, exact=True).click()


def switch(page, name):
    page.get_by_role('button', name='Switch profile:', exact=False).click()
    page.locator('.profile-option').filter(has_text=name).click()


def test_watchlist_sync_profile_isolation_and_reload(page):
    page.locator('.hero-actions').get_by_role('button', name='Add to Watchlist').click()
    page.get_by_role('button', name='View Details', exact=True).click()
    page.locator('.detail-copy').get_by_role('button', name='Remove from watchlist').wait_for()
    page.keyboard.press('Escape')
    nav(page, 'My List')
    page.locator('.movie-card').wait_for()
    assert page.locator('.movie-title').inner_text() == 'The Godfather'
    switch(page, 'Marcus Sterling')
    page.get_by_role('heading', name='Make room for great films.').wait_for()
    switch(page, 'Elena Vance')
    page.locator('.movie-card').wait_for()
    page.reload(wait_until='networkidle')
    nav(page, 'My List')
    page.locator('.movie-card').wait_for()
    page.get_by_role('button', name='Remove from watchlist').click()
    page.get_by_role('heading', name='Make room for great films.').wait_for()
    nav(page, 'Discover')
    page.locator('.hero-actions').get_by_role('button', name='Add to Watchlist').wait_for()


def test_search_filters_sort_list_and_empty_state(page):
    nav(page, 'Explore')
    page.get_by_text('3,416', exact=True).wait_for()
    search = page.get_by_role('textbox', name='Search catalog')
    search.fill('star')
    page.locator('.genre-filters').get_by_role('button', name='Sci-Fi', exact=True).click()
    page.get_by_label('Release era').select_option('1980')
    page.get_by_label('Sort movies').select_option('rating')
    page.wait_for_timeout(450)
    assert page.locator('.movie-card').count() == 7  # Star Wars and Star Trek in the 1980s.
    assert all('198' in x for x in page.locator('.movie-meta').all_inner_texts())
    page.get_by_role('button', name='List view').click()
    assert page.locator('.catalog-row').count() == 7
    search.fill('no-such-film-xyz')
    page.get_by_role('heading', name='No films in this frame.').wait_for()
    page.get_by_role('button', name='clear your filters').click()
    page.locator('.catalog-row').first.wait_for()
    assert page.get_by_label('Release era').input_value() == ''


def test_details_rating_dislike_escape_focus_and_id_namespace(page):
    trigger = page.get_by_role('button', name='View Details', exact=True)
    trigger.click()
    page.locator('.detail-copy h2').wait_for()
    for _ in range(12):
        page.keyboard.press('Tab')
        assert page.evaluate("!!document.activeElement.closest('dialog[open]')")
    page.get_by_role('button', name='Rate 4 stars', exact=True).click()
    assert json.loads(page.evaluate("localStorage.getItem('cinepulse_rated_12')"))['720'] == 4
    page.keyboard.press('Escape')
    assert page.locator('dialog[open]').count() == 0
    assert trigger.evaluate('(e) => document.activeElement === e')
    trigger.click()
    page.get_by_role('button', name='Rate 4 stars', exact=True).wait_for()
    assert page.get_by_role('button', name='Rate 4 stars', exact=True).get_attribute('aria-pressed') == 'true'
    page.get_by_role('button', name='Dislike film', exact=True).click()
    page.keyboard.press('Escape')
    assert page.locator('.hero h1').inner_text() != 'The Godfather'
    page.reload(wait_until='networkidle')
    assert page.locator('.hero h1').inner_text() != 'The Godfather'
    # 858 is a MovieLens ID, not the movie_idx namespace used by feedback.
    page.evaluate("localStorage.setItem('cinepulse_saved_12', '[858]')")
    page.reload(wait_until='networkidle')
    nav(page, 'My List')
    page.locator('.movie-title').wait_for()
    assert page.locator('.movie-title').inner_text() != 'The Godfather'


def test_onboarding_local_curation_and_keyboard_search(page):
    page.keyboard.press('Control+k')
    page.get_by_placeholder('Search by title, genre, or year', exact=True).fill('Matrix')
    page.locator('.search-result').first.wait_for()
    assert 'Matrix' in page.locator('.search-result').first.inner_text()
    page.keyboard.press('Escape')
    page.get_by_role('button', name='Switch profile:', exact=False).click()
    page.get_by_role('button', name='Create a profile', exact=True).click()
    page.get_by_label('What should we call you?').fill('Noir Fan')
    page.locator('.onboarding-genres').get_by_role('button', name='Film-Noir', exact=True).click()
    page.get_by_role('button', name='Start discovering', exact=True).click()
    page.locator('.hero h1').wait_for()
    page.get_by_role('heading', name='Curated For Your Profile', exact=True).wait_for()
    nav(page, 'Insights')
    page.get_by_text('No Two-Tower inference is performed for this profile.', exact=False).wait_for()
    assert page.locator('.donut').count() == 0
    assert page.get_by_text('Film-Noir', exact=True).count() == 1
    page.reload(wait_until='networkidle')
    assert 'Noir Fan' in page.get_by_role('button', name='Switch profile:', exact=False).get_attribute('aria-label')


def test_static_controls_and_verified_cohort(page):
    assert page.get_by_text('Precomputed V2 demo', exact=True).count() == 0
    assert page.locator('.discovery-intro, .demo-note').count() == 0
    assert 'Built with curiosity' not in page.locator('footer').inner_text()
    assert page.locator('footer a[href*="github.com"]').count() == 0
    assert page.get_by_text('Fine-tune live recommendations', exact=True).count() == 0
    assert page.get_by_text('Recalibrate Feed', exact=True).count() == 0
    nav(page, 'Benchmarks')
    text = page.locator('main').inner_text()
    for number in ['3,974', '0.0560', '0.0778', '18.09%', '22.19%', '95.08%', '27.92', '0.6933', '0.5165']:
        assert number in text
    assert page.locator('tbody tr').count() == 3
    assert page.locator('.experiment-grid article').count() == 6
