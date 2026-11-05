"""Repeatable Playwright visual review of the local static demo.

Run: .venv/Scripts/python.exe tests/capture_frontend.py pass1
"""
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path('artifacts/screenshots') / (sys.argv[1] if len(sys.argv) > 1 else 'final')
OUT.mkdir(parents=True, exist_ok=True)
SIZES = {'desktop': (1440, 900), 'mobile': (390, 844), 'tablet': (768, 1024), 'wide': (1920, 1080)}


def nav(page, label):
    page.locator('nav:visible').get_by_role('button', name=label, exact=True).click()
    page.wait_for_timeout(450)


def capture(page, name):
    page.wait_for_timeout(650)
    page.screenshot(path=str(OUT / f'{name}.png'))
    overflow = page.evaluate('document.documentElement.scrollWidth > innerWidth')
    assert not overflow, f'Horizontal page overflow: {name}'
    dialog = page.locator('dialog[open]')
    if dialog.count():
        assert dialog.evaluate('(e) => e.scrollWidth <= e.clientWidth + 1'), f'Dialog overflow: {name}'


def capture_full(page, name):
    # Load offscreen posters before the full-page review, rather than recording lazy placeholders.
    page.locator('img[loading="lazy"]').evaluate_all("els => els.forEach(e => e.loading = 'eager')")
    page.evaluate("Promise.race([Promise.all([...document.images].map(i => i.decode().catch(() => {}))), new Promise(r => setTimeout(r, 3500))])")
    page.screenshot(path=str(OUT / f'{name}-full.png'), full_page=True)


with sync_playwright() as p:
    browser = p.chromium.launch()
    errors = []
    for size, (width, height) in SIZES.items():
        context = browser.new_context(viewport={'width': width, 'height': height}, device_scale_factor=1, has_touch=size == 'mobile', is_mobile=size == 'mobile')
        page = context.new_page()
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto('http://localhost:8000', wait_until='networkidle')
        page.locator('.hero h1').wait_for()
        capture(page, f'{size}-discover')
        if size in ['desktop', 'mobile']:
            capture_full(page, f'{size}-discover')
        page.get_by_role('button', name='View Details', exact=True).click()
        page.locator('.detail-copy h2').wait_for()
        capture(page, f'{size}-details')
        page.keyboard.press('Escape')
        page.locator('.hero-actions').get_by_role('button', name='Add to Watchlist', exact=True).click()
        nav(page, 'My List')
        page.locator('.movie-card').first.wait_for()
        page.get_by_role('status').wait_for(state='hidden')
        capture(page, f'{size}-my-list')
        nav(page, 'Explore')
        page.locator('.movie-grid .movie-card').first.wait_for()
        capture(page, f'{size}-explore')
        if size in ['desktop', 'mobile']:
            capture_full(page, f'{size}-explore')
        nav(page, 'Insights')
        page.locator('.genre-bars').wait_for()
        capture(page, f'{size}-insights')
        if size in ['desktop', 'mobile']:
            capture_full(page, f'{size}-insights')
        nav(page, 'Benchmarks')
        capture(page, f'{size}-benchmarks')
        if size in ['desktop', 'mobile']:
            capture_full(page, f'{size}-benchmarks')
        page.get_by_role('button', name='Switch profile:', exact=False).click()
        capture(page, f'{size}-profiles')
        page.get_by_role('button', name='Create a profile', exact=True).click()
        capture(page, f'{size}-onboarding')
        context.close()
    browser.close()
    (OUT / 'browser-errors.json').write_text(json.dumps(errors, indent=2))
    assert not errors, errors
print(f'Captured all pages at four viewports: {OUT}')
