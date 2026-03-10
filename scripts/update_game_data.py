#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Game Data Update Script

Automates the update process for support cards, characters (trainees), and skills
from GameTora. Uses GameTora list pages to auto-discover new content.

Usage:
    # Auto-discover and add all new support cards and characters
    python scripts/update_game_data.py --discover

    # Auto-discover with dry-run (preview without modifying files)
    python scripts/update_game_data.py --discover --dry-run

    # Full automatic update (skills + discover + build)
    python scripts/update_game_data.py --all

    # Update skills (uses Selenium to scrape from skills page)
    python scripts/update_game_data.py --update-skills

    # Update skills with manual JSON URL (optional, skips Selenium)
    python scripts/update_game_data.py --update-skills --skills-url "https://gametora.com/data/umamusume/skills.XXXXXXXX.json"

    # List all existing support cards and characters
    python scripts/update_game_data.py --list

    # Add specific support cards (using GameTora slugs)
    python scripts/update_game_data.py --supports "30036-riko-kashimoto,30034-rice-shower"

    # Add specific characters/trainees
    python scripts/update_game_data.py --characters "105602-matikanefukukitaru"

    # Build only (after any updates)
    python scripts/update_game_data.py --build

    # Add cards with images and rebuild
    python scripts/update_game_data.py --supports "30036-riko-kashimoto" --images --build

    # Debug mode for troubleshooting
    python scripts/update_game_data.py --discover --debug
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import requests

# === Constants ===
BASE_URL = "https://gametora.com"
SKILLS_PAGE_URL = f"{BASE_URL}/umamusume/skills"
CHARACTERS_PAGE_URL = f"{BASE_URL}/umamusume/characters"
SUPPORTS_PAGE_URL = f"{BASE_URL}/umamusume/supports"

# Paths (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
IN_GAME_DIR = os.path.join(DATASETS_DIR, "in_game")
WEB_DIR = os.path.join(PROJECT_ROOT, "web")

SKILLS_JSON_PATH = os.path.join(IN_GAME_DIR, "skills.json")
EVENTS_JSON_PATH = os.path.join(IN_GAME_DIR, "events.json")
STATUS_JSON_PATH = os.path.join(IN_GAME_DIR, "status.json")
IMG_DIR = os.path.join(WEB_DIR, "public", "events")

# Regex to discover skills JSON URL from HTML
SKILLS_JSON_RX = re.compile(r'["\'](/?data/umamusume/skills\.[a-zA-Z0-9]+\.json)["\']')


def dbg(on: bool, *args, **kwargs):
    """Print debug messages to stderr."""
    if on:
        print(*args, file=sys.stderr, **kwargs)


def info(*args, **kwargs):
    """Print info messages."""
    print("[INFO]", *args, **kwargs)


def warn(*args, **kwargs):
    """Print warning messages."""
    print("[WARN]", *args, file=sys.stderr, **kwargs)


def error(*args, **kwargs):
    """Print error messages."""
    print("[ERROR]", *args, file=sys.stderr, **kwargs)


# === Skills Update ===

def fetch_skills_with_selenium(debug: bool = False) -> List[Dict[str, Any]]:
    """
    Fetch skills by using Selenium to capture the skills JSON data.
    The page loads skills via XHR, so we intercept that or find it in __NEXT_DATA__.
    """
    import time

    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.chrome.options import Options as ChromeOptions
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
    except ImportError as e:
        warn(f"Selenium not available: {e}")
        return []

    def dismiss_cookie_banner(driver) -> None:
        """Dismiss Quantcast cookie consent banner if present."""
        try:
            time.sleep(1)
            dismiss_js = """
            (function() {
                var qcButtons = document.querySelectorAll('.qc-cmp2-summary-buttons button, [class*="qc-cmp"] button');
                for (var btn of qcButtons) {
                    var text = (btn.textContent || btn.innerText || '').toLowerCase();
                    if (text.includes('agree') || text.includes('accept') || text.includes('consent')) {
                        btn.click();
                        return 'clicked qc button';
                    }
                }
                var allButtons = document.querySelectorAll('button, [role="button"]');
                for (var btn of allButtons) {
                    var text = (btn.textContent || btn.innerText || '').toLowerCase();
                    if (text.includes('agree') || text.includes('accept all') || text.includes('consent')) {
                        btn.click();
                        return 'clicked consent button';
                    }
                }
                var overlays = document.querySelectorAll('.qc-cmp-cleanslate, [class*="qc-cmp2"], .qc-cmp2-container');
                for (var overlay of overlays) {
                    overlay.remove();
                }
                return overlays.length > 0 ? 'removed overlay' : 'no banner found';
            })();
            """
            result = driver.execute_script(dismiss_js)
            dbg(debug, f"[DEBUG] Cookie banner dismiss result: {result}")
            time.sleep(1)
            driver.execute_script("""
                var overlays = document.querySelectorAll('.qc-cmp-cleanslate, [class*="qc-cmp2"]');
                overlays.forEach(function(el) { el.style.display = 'none'; });
            """)
            time.sleep(0.3)
        except Exception as e:
            dbg(debug, f"[DEBUG] Cookie banner dismiss error: {e}")

    def select_global_server(driver) -> None:
        """Select global server from the settings dropdown."""
        try:
            driver.find_element(
                By.CSS_SELECTOR,
                "#styles_page-header_alt__vF57o > div.styles_header-banner__OU9Yu > div > div > span"
            ).click()
            driver.find_element(
                By.CSS_SELECTOR,
                "#tippy-1 > div > div.tippy-content > div > div > div > div:nth-child(5) > label"
            ).click()
            time.sleep(2)
        except Exception as e:
            dbg(debug, f"[DEBUG] Server selection error (may already be global): {e}")

    def set_up(driver) -> None:
        driver.get(BASE_URL + "/umamusume")
        dismiss_cookie_banner(driver)
        select_global_server(driver)

    use_headless = os.environ.get("DEBUG_VISIBLE", "").lower() not in ("1", "true", "yes")

    try:
        dbg(debug, f"[DEBUG] Starting Chrome (headless={use_headless}) for skills scraping...")
        options = ChromeOptions()
        if use_headless:
            options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--window-size=1920,1080")

        # Enable performance logging to capture network requests
        options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

        driver = webdriver.Chrome(options=options)

        try:
            # Setup: dismiss cookie banner and select global server
            set_up(driver)

            # Navigate to skills page
            dbg(debug, f"[DEBUG] Loading skills page: {SKILLS_PAGE_URL}")
            driver.get(SKILLS_PAGE_URL)

            # Wait for page to load
            time.sleep(3)

            # Method 1: Try to find __NEXT_DATA__ which contains the skills data
            try:
                next_data_element = driver.find_element(By.ID, "__NEXT_DATA__")
                next_data_text = next_data_element.get_attribute("innerHTML")
                if next_data_text:
                    next_data = json.loads(next_data_text)
                    # Navigate to skills data in __NEXT_DATA__
                    skills_data = next_data.get("props", {}).get("pageProps", {}).get("skills", [])
                    if skills_data:
                        dbg(debug, f"[DEBUG] Found {len(skills_data)} skills in __NEXT_DATA__")
                        return parse_skills_from_json_data(skills_data, debug)
            except Exception as e:
                dbg(debug, f"[DEBUG] __NEXT_DATA__ extraction failed: {e}")

            # Method 2: Try to discover JSON URL from page source and fetch it
            page_source = driver.page_source
            match = SKILLS_JSON_RX.search(page_source)
            if match:
                path = match.group(1)
                json_url = path if path.startswith("http") else BASE_URL.rstrip("/") + "/" + path.lstrip("/")
                dbg(debug, f"[DEBUG] Found skills JSON URL in page: {json_url}")
                skills = fetch_skills_json(json_url, debug)
                if skills:
                    return skills

            # Method 3: Check network logs for skills JSON request
            try:
                logs = driver.get_log("performance")
                for log in logs:
                    message = json.loads(log["message"])["message"]
                    if message.get("method") == "Network.responseReceived":
                        url = message.get("params", {}).get("response", {}).get("url", "")
                        if "skills" in url and ".json" in url:
                            dbg(debug, f"[DEBUG] Found skills JSON URL in network logs: {url}")
                            skills = fetch_skills_json(url, debug)
                            if skills:
                                return skills
            except Exception as e:
                dbg(debug, f"[DEBUG] Network log extraction failed: {e}")

            warn("Could not extract skills data via Selenium")
            return []

        finally:
            driver.quit()

    except Exception as e:
        warn(f"Selenium skills scraping failed: {e}")
        return []


def parse_skills_from_json_data(raw: List[Dict[str, Any]], debug: bool = False) -> List[Dict[str, Any]]:
    """Parse skills from raw JSON data (from __NEXT_DATA__ or API response)."""
    # Rarity mapping
    rarity_map = {
        1: ("normal", "dnlGQR"),
        2: ("gold", "geDDHx"),
        3: ("unique", "bhlwbP"),
    }

    icon_base = f"{BASE_URL}/images/umamusume/skill_icons/"
    skills: List[Dict[str, Any]] = []

    for sk in raw:
        skill_id = sk.get("id")
        name_en = sk.get("name_en") or ""
        desc_en = sk.get("desc_en") or ""
        iconid = sk.get("iconid")
        rarity_code = sk.get("rarity")

        if skill_id is None or not name_en or not desc_en or iconid is None:
            continue

        if rarity_code in rarity_map:
            rarity, color_class = rarity_map[rarity_code]
        elif skill_id >= 900000:
            rarity, color_class = "inherited", "dnlGQR"
        else:
            rarity, color_class = "normal", "dnlGQR"

        icon_filename = f"utx_ico_skill_{iconid}.png"
        icon_src = icon_base + icon_filename

        # Grade symbol detection
        grade_symbol = None
        for sym in ("◎", "○"):
            if name_en.strip().endswith(sym):
                grade_symbol = sym
                break

        skills.append({
            "id": skill_id,
            "icon_filename": icon_filename,
            "icon_src": icon_src,
            "name": name_en,
            "description": desc_en,
            "color_class": color_class,
            "rarity": rarity,
            "grade_symbol": grade_symbol,
        })

    dbg(debug, f"[DEBUG] Parsed {len(skills)} skills from JSON data")
    return skills


def discover_skills_json_url(debug: bool = False) -> Optional[str]:
    """Auto-discover the skills JSON URL from GameTora HTML."""
    dbg(debug, f"[DEBUG] Fetching skills page: {SKILLS_PAGE_URL}")
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (UmaSkillScraper; +https://github.com/umaplay)",
            "Accept-Language": "en-US,en;q=0.9",
        }
        r = requests.get(SKILLS_PAGE_URL, headers=headers, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        error(f"Failed to fetch skills page: {e}")
        return None

    match = SKILLS_JSON_RX.search(r.text)
    if not match:
        dbg(debug, "[DEBUG] No skills JSON reference found in HTML.")
        return None

    path = match.group(1)
    url = path if path.startswith("http") else BASE_URL.rstrip("/") + "/" + path.lstrip("/")
    dbg(debug, f"[DEBUG] Discovered skills JSON URL: {url}")
    return url


def fetch_skills_json(url: str, debug: bool = False) -> List[Dict[str, Any]]:
    """Fetch and parse skills from JSON endpoint."""
    dbg(debug, f"[DEBUG] Fetching skills JSON: {url}")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        raw = r.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
        error(f"Failed to fetch/parse skills JSON: {e}")
        return []

    # Rarity mapping
    rarity_map = {
        1: ("normal", "dnlGQR"),
        2: ("gold", "geDDHx"),
        3: ("unique", "bhlwbP"),
    }

    icon_base = f"{BASE_URL}/images/umamusume/skill_icons/"
    skills: List[Dict[str, Any]] = []

    for sk in raw:
        skill_id = sk.get("id")
        name_en = sk.get("name_en") or ""
        desc_en = sk.get("desc_en") or ""
        iconid = sk.get("iconid")
        rarity_code = sk.get("rarity")

        if skill_id is None or not name_en or not desc_en or iconid is None:
            continue

        if rarity_code in rarity_map:
            rarity, color_class = rarity_map[rarity_code]
        elif skill_id >= 900000:
            rarity, color_class = "inherited", "dnlGQR"
        else:
            rarity, color_class = "normal", "dnlGQR"

        icon_filename = f"utx_ico_skill_{iconid}.png"
        icon_src = icon_base + icon_filename

        # Grade symbol detection
        grade_symbol = None
        for sym in ("◎", "○"):
            if name_en.strip().endswith(sym):
                grade_symbol = sym
                break

        skills.append({
            "id": skill_id,
            "icon_filename": icon_filename,
            "icon_src": icon_src,
            "name": name_en,
            "description": desc_en,
            "color_class": color_class,
            "rarity": rarity,
            "grade_symbol": grade_symbol,
        })

    dbg(debug, f"[DEBUG] Parsed {len(skills)} skills from JSON")
    return skills


def update_skills(
    skills_url: Optional[str] = None,
    debug: bool = False,
    dry_run: bool = False,
) -> bool:
    """Auto-discover and update skills.json from GameTora using Selenium."""
    info("Updating skills...")

    skills: List[Dict[str, Any]] = []

    # If a manual URL is provided, try JSON mode first
    if skills_url:
        info(f"Using provided skills JSON URL: {skills_url}")
        skills = fetch_skills_json(skills_url, debug)

    # Default to Selenium scraping (same as characters/support cards)
    if not skills:
        info("Scraping skills from GameTora using Selenium...")
        skills = fetch_skills_with_selenium(debug)

    if not skills:
        error("Could not fetch skills via Selenium.")
        error("Alternatively, you can provide a manual JSON URL with --skills-url")
        return False

    if dry_run:
        info(f"[DRY RUN] Would update {SKILLS_JSON_PATH} with {len(skills)} skills")
        return True

    # Backup existing file
    if os.path.exists(SKILLS_JSON_PATH):
        backup_path = SKILLS_JSON_PATH + ".bak"
        shutil.copy2(SKILLS_JSON_PATH, backup_path)
        dbg(debug, f"[DEBUG] Created backup: {backup_path}")

    with open(SKILLS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(skills, f, ensure_ascii=False, indent=2)

    info(f"Updated {SKILLS_JSON_PATH} with {len(skills)} skills")
    return True


# === Auto-Discovery ===

def load_existing_events(debug: bool = False) -> List[Dict[str, Any]]:
    """Load existing events.json."""
    if not os.path.exists(EVENTS_JSON_PATH):
        return []
    try:
        with open(EVENTS_JSON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        dbg(debug, f"[DEBUG] Loaded {len(data)} existing entries from events.json")
        return data
    except (json.JSONDecodeError, IOError) as e:
        warn(f"Failed to load events.json: {e}")
        return []


def list_existing_cards(debug: bool = False) -> None:
    """List all existing support cards and characters in events.json."""
    existing = load_existing_events(debug)

    supports = [e for e in existing if e.get("type") == "support"]
    trainees = [e for e in existing if e.get("type") == "trainee"]

    info(f"Existing support cards ({len(supports)}):")
    for s in sorted(supports, key=lambda x: x.get("name", "").lower()):
        info(f"  - {s.get('name')} ({s.get('rarity', '?')}/{s.get('attribute', '?')})")

    info(f"\nExisting characters/trainees ({len(trainees)}):")
    for t in sorted(trainees, key=lambda x: x.get("name", "").lower()):
        info(f"  - {t.get('name')}")


def fetch_slugs_with_selenium(
    url: str,
    pattern: re.Pattern[str],
    debug: bool = False,
    page_kind: str = "generic",
) -> List[str]:
    """Render a list page with Selenium and extract slugs from anchor hrefs."""
    import time

    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.edge.options import Options as EdgeOptions
        from selenium.webdriver.chrome.options import Options as ChromeOptions
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
    except Exception as e:
        warn(f"Selenium not available for rendered extraction: {e}")
        return []

    def dismiss_cookie_banner(driver) -> None:
        """Dismiss Quantcast cookie consent banner if present."""
        try:
            # Wait a moment for cookie banner to appear
            time.sleep(1)

            # Method 1: Try to find and click Quantcast consent button
            # The Quantcast banner typically has buttons with specific text
            dismiss_js = """
            (function() {
                // Try Quantcast-specific buttons first
                var qcButtons = document.querySelectorAll('.qc-cmp2-summary-buttons button, [class*="qc-cmp"] button');
                for (var btn of qcButtons) {
                    var text = (btn.textContent || btn.innerText || '').toLowerCase();
                    if (text.includes('agree') || text.includes('accept') || text.includes('consent')) {
                        btn.click();
                        return 'clicked qc button';
                    }
                }

                // Try generic consent buttons
                var allButtons = document.querySelectorAll('button, [role="button"]');
                for (var btn of allButtons) {
                    var text = (btn.textContent || btn.innerText || '').toLowerCase();
                    if (text.includes('agree') || text.includes('accept all') || text.includes('consent')) {
                        btn.click();
                        return 'clicked consent button';
                    }
                }

                // Method 2: Force remove the overlay if click didn't work
                var overlays = document.querySelectorAll('.qc-cmp-cleanslate, [class*="qc-cmp2"], .qc-cmp2-container');
                for (var overlay of overlays) {
                    overlay.remove();
                }
                if (overlays.length > 0) {
                    return 'removed overlay';
                }

                return 'no banner found';
            })();
            """
            result = driver.execute_script(dismiss_js)
            dbg(debug, f"[DEBUG] Cookie banner dismiss result: {result}")
            time.sleep(1)

            # Double-check: if overlay still exists, force remove it
            driver.execute_script("""
                var overlays = document.querySelectorAll('.qc-cmp-cleanslate, [class*="qc-cmp2"]');
                overlays.forEach(function(el) { el.style.display = 'none'; });
            """)
            time.sleep(0.3)
        except Exception as e:
            dbg(debug, f"[DEBUG] Cookie banner dismiss error: {e}")


    def select_global_server(driver) -> None:
        # Finds and clicks the server setting button
        driver.find_element(By.CSS_SELECTOR, "#styles_page-header_alt__vF57o > div.styles_header-banner__OU9Yu > div > div > span"
        ).click()
        # Selects the global server
        driver.find_element(By.CSS_SELECTOR, "#tippy-1 > div > div.tippy-content > div > div > div > div:nth-child(5) > label").click()
        time.sleep(2)

    def set_up(driver) -> None:
        driver.get(BASE_URL + "/umamusume")
        dismiss_cookie_banner(driver)
        select_global_server(driver)

    def collect_with_driver(driver) -> List[str]:
        # First, set up driver by dismissing cookie dialog, and selecting global server
        set_up(driver)

        dbg(debug, f"[DEBUG] Loading page: {url}")
        driver.get(url)

        # Wait for page to load
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/umamusume/']"))
            )
        except Exception:
            dbg(debug, "[DEBUG] Timeout waiting for page links, continuing anyway...")

        # Debug: Check what the page looks like
        if debug:
            try:
                page_title = driver.title
                dbg(debug, f"[DEBUG] Page title: {page_title}")
                # Check if settings icon exists
                settings_imgs = driver.find_elements(By.CSS_SELECTOR, "img[src*='settings']")
                dbg(debug, f"[DEBUG] Found {len(settings_imgs)} settings images")
                for img in settings_imgs:
                    dbg(debug, f"[DEBUG]   - src: {img.get_attribute('src')}")
            except Exception as e:
                dbg(debug, f"[DEBUG] Debug check failed: {e}")

        # Collect all hrefs, skipping hidden/unreleased entries.
        slugs = set()
        for el in driver.find_elements(By.CSS_SELECTOR, "a[href]"):
            try:
                if el.get_attribute("hidden") is not None:
                    continue
                if el.find_elements(By.CSS_SELECTOR, "[hidden]"):
                    continue
            except Exception:
                pass

            href = el.get_attribute("href")
            if not href:
                continue
            match = pattern.search(href.lower())
            if match:
                slugs.add(match.group(1))
        dbg(debug, f"[DEBUG] Found {len(slugs)} unique slugs")
        return sorted(slugs)

    # Use Chrome only (remove Edge option)
    # Use non-headless for debugging if DEBUG_VISIBLE env var is set
    use_headless = os.environ.get("DEBUG_VISIBLE", "").lower() not in ("1", "true", "yes")

    last_error: Optional[Exception] = None
    for browser in ("chrome",):
        try:
            dbg(debug, f"[DEBUG] Trying {browser} browser (headless={use_headless})...")
            options = ChromeOptions()
            if use_headless:
                options.add_argument("--headless")
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--window-size=1920,1080")
            driver = webdriver.Chrome(options=options)

            try:
                slugs = collect_with_driver(driver)
                info(f"Selenium ({browser}) extracted {len(slugs)} {page_kind} slugs")
                return slugs
            finally:
                driver.quit()
        except Exception as e:
            last_error = e
            dbg(debug, f"[DEBUG] Selenium {browser} failed: {e}")

    warn(f"Headless extraction failed for {url}: {last_error}")
    return []


def get_existing_names(debug: bool = False) -> Tuple[set, set]:
    """
    Get sets of existing support and character names from events.json.
    Returns (support_names, character_names) as lowercase sets.
    """
    existing = load_existing_events(debug)

    support_names = set()
    for e in existing:
        if e.get("type") == "support":
            name = e.get("name", "").lower().strip()
            if name:
                support_names.add(name)

    char_names = set()
    for e in existing:
        if e.get("type") == "trainee":
            name = e.get("name", "").lower().strip()
            if name:
                char_names.add(name)
                # Also add base name without version suffix for matching
                # e.g., "Special Week (Summer)" -> "special week"
                base_name = re.sub(r'\s*\([^)]+\)\s*$', '', name).strip()
                if base_name:
                    char_names.add(base_name)

    return support_names, char_names


def slug_to_name(slug: str) -> str:
    """Convert a slug like '30036-riko-kashimoto' to 'Riko Kashimoto'."""
    # Remove leading ID
    name_part = re.sub(r'^\d+-', '', slug)
    # Convert hyphens to spaces and title case
    return name_part.replace('-', ' ').title()


def discover_new_supports(debug: bool = False) -> List[str]:
    """
    Discover new support cards by comparing GameTora's sitemap with events.json.
    Returns list of slugs for new support cards.
    """
    info("Discovering new support cards from GameTora list page...")

    support_pattern = re.compile(r'/umamusume/supports/(\d{5}-[a-z0-9-]+)', re.I)
    page_supports = fetch_slugs_with_selenium(
        SUPPORTS_PAGE_URL,
        support_pattern,
        debug=debug,
        page_kind="supports",
    )
    if not page_supports:
        warn("Could not fetch supports page or no supports found")
        return []

    # Do not filter by name because multiple cards share the same name (different rarities/attributes).
    new_slugs = list(page_supports)
    info(f"Found {len(new_slugs)} support card slug(s) out of {len(page_supports)} total")

    if new_slugs and debug:
        info("New support slugs:")
        for slug in new_slugs[:20]:
            info(f"  - {slug}")
        if len(new_slugs) > 20:
            info(f"  ... and {len(new_slugs) - 20} more")

    return new_slugs


def discover_new_characters(debug: bool = False) -> List[str]:
    """
    Discover new characters by comparing GameTora's sitemap with events.json.
    Returns list of slugs for new characters.
    """
    info("Discovering new characters from GameTora list page...")

    char_pattern = re.compile(r'/umamusume/characters/(\d{6}-[a-z0-9-]+)', re.I)
    page_chars = fetch_slugs_with_selenium(
        CHARACTERS_PAGE_URL,
        char_pattern,
        debug=debug,
        page_kind="characters",
    )
    if not page_chars:
        warn("Could not fetch characters page or no characters found")
        return []

    # Do not filter by name because multiple versions share the same base name.
    new_slugs = list(page_chars)
    info(f"Found {len(new_slugs)} character slug(s) out of {len(page_chars)} total")

    if new_slugs and debug:
        info("New character slugs:")
        for slug in new_slugs[:20]:
            info(f"  - {slug}")
        if len(new_slugs) > 20:
            info(f"  ... and {len(new_slugs) - 20} more")

    return new_slugs


# === Event Fetching and Merging ===

def fetch_events_with_scraper(
    support_slugs: List[str],
    character_slugs: List[str],
    download_images: bool = False,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """
    Fetch events using the existing scrape_events.py scraper.
    Returns list of parsed event entries.
    """
    if not support_slugs and not character_slugs:
        return []

    scraper_path = os.path.join(DATASETS_DIR, "scrape_events.py")
    if not os.path.exists(scraper_path):
        error(f"Scraper not found: {scraper_path}")
        return []

    # Build command
    cmd = [
        sys.executable,
        scraper_path,
        "--skills", SKILLS_JSON_PATH,
        "--status", STATUS_JSON_PATH,
        "--out", os.path.join(DATASETS_DIR, "_temp_events.json"),
    ]

    if support_slugs:
        cmd.extend(["--supports-card", ",".join(support_slugs)])

    if character_slugs:
        cmd.extend(["--characters-card", ",".join(character_slugs)])

    if download_images:
        cmd.extend(["--images", "--img-dir", IMG_DIR])

    if debug:
        cmd.append("--debug")

    dbg(debug, f"[DEBUG] Running scraper: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=DATASETS_DIR)
        if result.returncode != 0:
            error(f"Scraper failed: {result.stderr}")
            return []
        dbg(debug, f"[DEBUG] Scraper output: {result.stdout}")
    except Exception as e:
        error(f"Failed to run scraper: {e}")
        return []

    # Load results
    temp_path = os.path.join(DATASETS_DIR, "_temp_events.json")
    if not os.path.exists(temp_path):
        error("Scraper did not produce output file")
        return []

    try:
        with open(temp_path, "r", encoding="utf-8") as f:
            entries = json.load(f)
        os.remove(temp_path)  # Clean up
        return entries
    except (json.JSONDecodeError, IOError) as e:
        error(f"Failed to load scraper output: {e}")
        return []


def create_entry_key(entry: Dict[str, Any]) -> str:
    """Create unique key for an event entry.

    Uses gametora_id when available to distinguish between different versions
    of cards that share the same name, attribute, and rarity (e.g., two SSR Speed
    Silence Suzuka cards).
    """
    if entry.get("type") == "support":
        # Use gametora_id if available for unique identification
        gametora_id = entry.get("gametora_id")
        if gametora_id:
            return f"gt_{gametora_id}".lower()
        # Fallback for legacy entries without gametora_id
        return f"{entry.get('name', '')}_{entry.get('attribute', '')}_{entry.get('rarity', '')}".lower()
    else:  # trainee
        gametora_id = entry.get("gametora_id")
        if gametora_id:
            return f"gt_{gametora_id}".lower()
        return f"{entry.get('name', '')}_profile".lower()


def _legacy_key(entry: Dict[str, Any]) -> str:
    """Create legacy key (name/attribute/rarity) for migration compatibility."""
    if entry.get("type") == "support":
        return f"{entry.get('name', '')}_{entry.get('attribute', '')}_{entry.get('rarity', '')}".lower()
    else:
        return f"{entry.get('name', '')}_profile".lower()


def merge_events(
    existing: List[Dict[str, Any]],
    new_entries: List[Dict[str, Any]],
    debug: bool = False,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Merge new entries into existing events.
    Returns (merged_list, added_count, updated_count).

    Handles migration from old entries (without gametora_id) to new entries (with gametora_id).
    When a new entry with gametora_id matches an old entry by name/attribute/rarity,
    the old entry is replaced.
    """
    # Build lookup by key
    lookup: Dict[str, int] = {}
    for idx, entry in enumerate(existing):
        key = create_entry_key(entry)
        lookup[key] = idx

    # Also build a legacy lookup for entries without gametora_id (for migration)
    legacy_lookup: Dict[str, int] = {}
    for idx, entry in enumerate(existing):
        if not entry.get("gametora_id"):
            legacy_key = _legacy_key(entry)
            legacy_lookup[legacy_key] = idx

    # Track which legacy entries have been replaced (to avoid double-replacement)
    replaced_legacy_indices: set = set()

    merged = list(existing)
    added = 0
    updated = 0

    for new_entry in new_entries:
        key = create_entry_key(new_entry)
        if key in lookup:
            # Update existing entry with same key
            idx = lookup[key]
            merged[idx] = new_entry
            updated += 1
            dbg(debug, f"[DEBUG] Updated existing entry: {key}")
        else:
            # Check if this new entry (with gametora_id) should replace a legacy entry
            new_has_gametora_id = bool(new_entry.get("gametora_id"))
            legacy_key = _legacy_key(new_entry)

            if new_has_gametora_id and legacy_key in legacy_lookup:
                legacy_idx = legacy_lookup[legacy_key]
                if legacy_idx not in replaced_legacy_indices:
                    # Replace the legacy entry with the new one
                    merged[legacy_idx] = new_entry
                    replaced_legacy_indices.add(legacy_idx)
                    lookup[key] = legacy_idx
                    updated += 1
                    dbg(debug, f"[DEBUG] Replaced legacy entry '{legacy_key}' with new entry: {key}")
                else:
                    # Legacy entry already replaced by another card with same name/attr/rarity
                    # This is the second (or more) card with same name/attr/rarity - add as new
                    merged.append(new_entry)
                    lookup[key] = len(merged) - 1
                    added += 1
                    dbg(debug, f"[DEBUG] Added additional entry (same name/attr/rarity): {key}")
            else:
                # Add new entry
                merged.append(new_entry)
                lookup[key] = len(merged) - 1
                added += 1
                dbg(debug, f"[DEBUG] Added new entry: {key}")

    return merged, added, updated


def sort_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort events by type (support first, then trainee), then by name."""
    def sort_key(entry: Dict[str, Any]) -> Tuple[int, str, str, str]:
        type_order = 0 if entry.get("type") == "support" else 1
        return (
            type_order,
            entry.get("name", "").lower(),
            entry.get("rarity", "").lower(),
            entry.get("attribute", "").lower(),
        )
    return sorted(events, key=sort_key)


def update_events(
    support_slugs: List[str],
    character_slugs: List[str],
    download_images: bool = False,
    dry_run: bool = False,
    debug: bool = False,
) -> Tuple[int, int]:
    """
    Fetch events and merge into events.json.
    Returns (added, updated) counts.
    """
    if not support_slugs and not character_slugs:
        info("No slugs to process")
        return 0, 0

    info(f"Fetching events for {len(support_slugs)} supports, {len(character_slugs)} characters...")

    new_entries = fetch_events_with_scraper(
        support_slugs, character_slugs, download_images, debug
    )

    if not new_entries:
        warn("No events fetched")
        return 0, 0

    existing = load_existing_events(debug)
    merged, added, updated = merge_events(existing, new_entries, debug)

    if dry_run:
        info(f"[DRY RUN] Would add {added} entries, update {updated} entries")
        return added, updated

    # Sort before writing
    merged = sort_events(merged)

    # Backup existing file
    if os.path.exists(EVENTS_JSON_PATH):
        backup_path = EVENTS_JSON_PATH + ".bak"
        shutil.copy2(EVENTS_JSON_PATH, backup_path)
        dbg(debug, f"[DEBUG] Created backup: {backup_path}")

    # Write merged data
    with open(EVENTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent="\t")

    info(f"Updated {EVENTS_JSON_PATH}: {added} added, {updated} updated (total: {len(merged)})")
    return added, updated


# === Build ===

def build_assets(debug: bool = False, dry_run: bool = False) -> bool:
    """Run build_catalog.py and npm run build."""
    info("Building assets...")

    if dry_run:
        info("[DRY RUN] Would run build_catalog.py and npm run build")
        return True

    # Run build_catalog.py
    catalog_script = os.path.join(PROJECT_ROOT, "build_catalog.py")
    if os.path.exists(catalog_script):
        info("  Running build_catalog.py...")
        try:
            result = subprocess.run(
                [sys.executable, catalog_script],
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            if result.returncode != 0:
                error(f"build_catalog.py failed: {result.stderr}")
                return False
            dbg(debug, f"[DEBUG] build_catalog output: {result.stdout}")
        except Exception as e:
            error(f"Failed to run build_catalog.py: {e}")
            return False
    else:
        warn(f"build_catalog.py not found at {catalog_script}")

    # Run npm run build
    if os.path.exists(WEB_DIR):
        info("  Running npm run build...")
        try:
            result = subprocess.run(
                ["npm", "run", "build"],
                capture_output=True,
                text=True,
                cwd=WEB_DIR,
                shell=True,  # Required on Windows
            )
            if result.returncode != 0:
                error(f"npm run build failed: {result.stderr}")
                return False
            dbg(debug, f"[DEBUG] npm build output: {result.stdout}")
        except Exception as e:
            error(f"Failed to run npm build: {e}")
            return False
    else:
        warn(f"Web directory not found at {WEB_DIR}")

    info("Build completed successfully")
    return True


# === CLI ===

def main():
    parser = argparse.ArgumentParser(
        description="Unified game data update script for Umaplay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode flags
    parser.add_argument(
        "--update-skills",
        action="store_true",
        help="Update skills.json from GameTora (auto-discovers JSON URL)",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Auto-discover new support cards and characters from GameTora list pages",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all existing support cards and characters in events.json",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Run build steps (build_catalog.py + npm run build)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all steps: skills update, auto-discovery, and build",
    )

    # Manual mode
    parser.add_argument(
        "--supports",
        type=str,
        help="Comma-separated GameTora support slugs (e.g., '30036-riko-kashimoto,30034-rice-shower')",
    )
    parser.add_argument(
        "--characters",
        type=str,
        help="Comma-separated GameTora character slugs (e.g., '105602-matikanefukukitaru')",
    )

    # Skills options
    parser.add_argument(
        "--skills-url",
        type=str,
        metavar="URL",
        help="Manual skills JSON URL (e.g., https://gametora.com/data/umamusume/skills.XXXXXXXX.json)",
    )

    # Options
    parser.add_argument(
        "--images",
        action="store_true",
        help="Download card images during event fetching",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying any files",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug output",
    )

    args = parser.parse_args()

    # Validate arguments
    if not any([args.update_skills, args.discover, args.build, args.all, args.supports, args.characters, args.list]):
        parser.print_help()
        sys.exit(1)

    if args.dry_run:
        info("[DRY RUN MODE] No files will be modified")

    success = True

    # === List existing cards ===
    if args.list:
        list_existing_cards(args.debug)
        if not any([args.update_skills, args.discover, args.build, args.all, args.supports, args.characters]):
            sys.exit(0)

    # === Skills Update ===
    if args.update_skills or args.all:
        if not update_skills(args.skills_url, args.debug, args.dry_run):
            success = False

    # === Auto-Discovery ===
    support_slugs: List[str] = []
    character_slugs: List[str] = []

    if args.supports:
        support_slugs.extend([s.strip() for s in args.supports.split(",") if s.strip()])

    if args.characters:
        character_slugs.extend([s.strip() for s in args.characters.split(",") if s.strip()])

    if args.discover or args.all:
        # Auto-discover new cards from GameTora sitemap
        discovered_supports = discover_new_supports(debug=args.debug)
        discovered_characters = discover_new_characters(debug=args.debug)
        support_slugs.extend(discovered_supports)
        character_slugs.extend(discovered_characters)

    # === Event Fetching ===
    if support_slugs or character_slugs:
        added, updated = update_events(
            support_slugs,
            character_slugs,
            download_images=args.images,
            dry_run=args.dry_run,
            debug=args.debug,
        )
        if added == 0 and updated == 0 and (support_slugs or character_slugs):
            success = False

    # === Build ===
    if args.build or args.all:
        if not build_assets(args.debug, args.dry_run):
            success = False

    # Summary
    if success:
        info("All operations completed successfully")
    else:
        error("Some operations failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
