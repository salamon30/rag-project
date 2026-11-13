"""Open the hosted demo in a real browser so Streamlit Community Cloud keeps it running.

A free Streamlit app goes to sleep when nobody visits it; the next visitor then sees a
"wake up" page and waits about a minute. A plain HTTP request does not count as a visit,
because the app only runs once a browser opens its websocket. So this script loads the
page in headless Chromium and, if the app is asleep, presses the wake-up button.

Run by .github/workflows/keep-demo-awake.yml. Locally:
    pip install playwright && playwright install chromium
    python scripts/keep_demo_awake.py
"""
import os
import sys

from playwright.sync_api import sync_playwright

URL = os.getenv("DEMO_URL", "https://medical-ai-papers.streamlit.app/")
WAKE_BUTTON = "Yes, get this app back up"
APP_TEXT = "Your question"  # label of the app's input box, only there once it runs
TIMEOUT_MS = 240_000  # a cold start loads the models and can take a few minutes


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=os.getenv("BROWSER_CHANNEL") or None)
        page = browser.new_page()
        page.goto(URL, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
        button = page.get_by_role("button", name=WAKE_BUTTON)
        try:
            button.wait_for(state="visible", timeout=15_000)
            print("The app was asleep. Pressing the wake-up button.")
            button.click()
        except Exception:
            print("No wake-up button: the app is running or already starting.")
        # The app renders inside an iframe on *.streamlit.app, so look in every frame.
        deadline = TIMEOUT_MS
        while deadline > 0:
            if any(APP_TEXT in (f.locator("body").inner_text(timeout=5_000) if f.locator("body").count() else "")
                   for f in page.frames):
                print("The app is up.")
                browser.close()
                return 0
            page.wait_for_timeout(5_000)
            deadline -= 5_000
        print("The app did not come up in time.", file=sys.stderr)
        browser.close()
        return 1


if __name__ == "__main__":
    sys.exit(main())
