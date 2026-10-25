#!/usr/bin/env python3
"""Check rendered Mermaid diagrams in a built guide (requires Playwright Chromium)."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('html_dir', type=Path)
    args = parser.parse_args()
    root = args.html_dir.resolve()
    pages = [p for p in root.rglob('*.html') if 'class="mermaid"' in p.read_text()]
    if not pages:
        raise SystemExit('No Mermaid pages found; build the guide first.')
    server = ThreadingHTTPServer(('127.0.0.1', 0), partial(SimpleHTTPRequestHandler, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                page = browser.new_page()
                for source in pages:
                    errors = []
                    capture_error = lambda error: errors.append(error)
                    page.on('pageerror', capture_error)
                    page.goto(f'http://127.0.0.1:{server.server_port}/{source.relative_to(root)}', wait_until='load')
                    page.wait_for_function("""() => [...document.querySelectorAll('.mermaid')].every(e => e.querySelector('svg'))""")
                    diagrams = page.locator('.mermaid')
                    for diagram in diagrams.all():
                        assert diagram.locator('.error-text, .error-icon').count() == 0, f'{source}: Mermaid error graphic'
                        assert diagram.locator('svg g.node').count() > 0, f'{source}: no flowchart nodes rendered'
                        assert diagram.locator('svg .edgePaths path').count() > 0, f'{source}: no flowchart connectors rendered'
                    assert not errors, f'{source}: browser errors: {errors}'
                    page.remove_listener('pageerror', capture_error)
                    print(f'PASS: {source.relative_to(root)}: {diagrams.count()} Mermaid diagrams rendered without errors')
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
