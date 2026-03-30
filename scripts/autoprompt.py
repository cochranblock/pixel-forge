#!/usr/bin/env python3
"""
Pixel Forge Auto-Prompter — drives two Firefox/Gemini sessions in parallel.

Usage:
    1. Open two Firefox windows, each logged into a different Gemini account
    2. Note their geckodriver ports (default 4444 and 4445)
    3. Run: python3 scripts/autoprompt.py

Requires: pip3 install selenium
"""

import os
import sys
import time
import json
import glob
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.firefox.service import Service
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
except ImportError:
    print("Installing selenium...")
    subprocess.run([sys.executable, "-m", "pip", "install", "selenium"], check=True)
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.firefox.options import Options
    from selenium.webdriver.firefox.service import Service
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

GEMINI_URL = "https://gemini.google.com/app"
OUTPUT_DIR = "data/raw/gemini"
PROMPT_FILE = "scripts/gemini-prompts.md"

def load_prompts(filepath):
    """Extract prompts from the markdown file."""
    prompts = []
    current_label = None
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("### "):
                # Extract label from "### Prompt N — label" or "### DND-N — label"
                parts = line.split("—")
                if len(parts) >= 2:
                    current_label = parts[1].strip().lower().replace(" ", "_")
                else:
                    current_label = line.split()[-1].lower()
            elif line.startswith("Create a ") and current_label:
                prompts.append({"label": current_label, "prompt": line})
                current_label = None
    return prompts

def create_driver(profile_path=None, port=4444):
    """Create a Firefox WebDriver using an existing profile."""
    options = Options()
    if profile_path:
        options.profile = profile_path

    service = Service(port=port)
    driver = webdriver.Firefox(service=service, options=options)
    return driver

def find_firefox_profiles():
    """Find Firefox profiles on macOS."""
    profiles_dir = Path.home() / "Library" / "Application Support" / "Firefox" / "Profiles"
    if not profiles_dir.exists():
        return []
    profiles = []
    for p in profiles_dir.iterdir():
        if p.is_dir():
            profiles.append(str(p))
    return sorted(profiles)

def send_prompt_and_download(driver, prompt_text, label, index, output_dir):
    """Send a prompt to Gemini and download the generated image."""
    try:
        # Find the text input area
        # Gemini's input is typically a contenteditable div or textarea
        # This may need adjustment as Gemini's UI changes
        input_selectors = [
            "div.ql-editor",                    # Quill editor
            "textarea",                          # Plain textarea
            "[contenteditable='true']",          # Content editable div
            ".text-input-field",                 # Generic class
            "div[role='textbox']",              # ARIA textbox
        ]

        input_el = None
        for selector in input_selectors:
            try:
                input_el = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                if input_el:
                    break
            except:
                continue

        if not input_el:
            print(f"  [{index}] Could not find input field")
            return False

        # Clear and type the prompt
        input_el.click()
        time.sleep(0.5)
        input_el.send_keys(prompt_text)
        time.sleep(0.5)

        # Submit — try Enter key or find submit button
        try:
            submit_btn = driver.find_element(By.CSS_SELECTOR, "button[aria-label='Send message']")
            submit_btn.click()
        except:
            input_el.send_keys(Keys.RETURN)

        # Wait for image generation (can take 30-120 seconds)
        print(f"  [{index}] Waiting for generation: {label}...")

        # Wait for an image to appear in the response
        img_el = None
        for attempt in range(60):  # Wait up to 5 minutes
            time.sleep(5)
            try:
                # Look for generated images
                images = driver.find_elements(By.CSS_SELECTOR, "img[src*='blob:'], img[src*='data:image'], img.generated-image")
                if images:
                    img_el = images[-1]  # Get the latest image
                    break
            except:
                pass

        if not img_el:
            print(f"  [{index}] No image generated for {label}")
            return False

        # Download the image
        img_src = img_el.get_attribute("src")
        output_path = os.path.join(output_dir, f"auto_{label}_{index:04d}")

        if img_src.startswith("data:image"):
            # Base64 encoded — decode and save
            import base64
            header, data = img_src.split(",", 1)
            ext = "png" if "png" in header else "jpg"
            with open(f"{output_path}.{ext}", "wb") as f:
                f.write(base64.b64decode(data))
        elif img_src.startswith("blob:"):
            # Blob URL — need to fetch via JS
            script = """
            var img = arguments[0];
            var canvas = document.createElement('canvas');
            canvas.width = img.naturalWidth;
            canvas.height = img.naturalHeight;
            canvas.getContext('2d').drawImage(img, 0, 0);
            return canvas.toDataURL('image/png').split(',')[1];
            """
            b64 = driver.execute_script(script, img_el)
            if b64:
                import base64
                with open(f"{output_path}.png", "wb") as f:
                    f.write(base64.b64decode(b64))
        else:
            # Regular URL — download via requests
            import urllib.request
            ext = "png" if "png" in img_src else "jpg"
            urllib.request.urlretrieve(img_src, f"{output_path}.{ext}")

        print(f"  [{index}] Saved: {label}")
        return True

    except Exception as e:
        print(f"  [{index}] Error: {e}")
        return False

def run_worker(profile_path, prompts, start_index, output_dir, port):
    """Run a single browser worker through its assigned prompts."""
    print(f"Starting worker on port {port} with {len(prompts)} prompts")

    driver = create_driver(profile_path, port)
    driver.get(GEMINI_URL)

    # Wait for page to load and user to be logged in
    print(f"Worker {port}: Navigate to Gemini. Waiting 10s for page load...")
    time.sleep(10)

    completed = 0
    for i, prompt_data in enumerate(prompts):
        idx = start_index + i
        success = send_prompt_and_download(
            driver, prompt_data["prompt"], prompt_data["label"], idx, output_dir
        )
        if success:
            completed += 1

        # Rate limit — wait between prompts
        time.sleep(5)

    print(f"Worker {port}: Completed {completed}/{len(prompts)}")
    driver.quit()
    return completed

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load prompts
    prompts = load_prompts(PROMPT_FILE)
    print(f"Loaded {len(prompts)} prompts from {PROMPT_FILE}")

    # Check how many are already done
    existing = len(glob.glob(os.path.join(OUTPUT_DIR, "auto_*")))
    if existing > 0:
        print(f"Found {existing} existing auto-generated files, skipping those prompts")
        prompts = prompts[existing:]

    if not prompts:
        print("All prompts already completed!")
        return

    # Find Firefox profiles
    profiles = find_firefox_profiles()
    print(f"Found {len(profiles)} Firefox profiles:")
    for i, p in enumerate(profiles):
        print(f"  {i}: {os.path.basename(p)}")

    # Split prompts between two workers
    mid = len(prompts) // 2
    batch1 = prompts[:mid]
    batch2 = prompts[mid:]

    print(f"\nWorker 1: {len(batch1)} prompts (port 4444)")
    print(f"Worker 2: {len(batch2)} prompts (port 4445)")
    print(f"\nStarting in 5 seconds... Make sure both Firefox windows are logged into Gemini.")
    time.sleep(5)

    # Use first two profiles, or default if not enough
    p1 = profiles[0] if len(profiles) > 0 else None
    p2 = profiles[1] if len(profiles) > 1 else None

    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(run_worker, p1, batch1, existing, OUTPUT_DIR, 4444)
        f2 = executor.submit(run_worker, p2, batch2, existing + mid, OUTPUT_DIR, 4445)

        for f in as_completed([f1, f2]):
            try:
                count = f.result()
                print(f"Worker finished: {count} sprites")
            except Exception as e:
                print(f"Worker failed: {e}")

    # Count total
    total = len(glob.glob(os.path.join(OUTPUT_DIR, "*")))
    print(f"\nDone. Total files in {OUTPUT_DIR}: {total}")
    print(f"Run 'pixel-forge ingest-gemini' to process them.")

if __name__ == "__main__":
    main()
