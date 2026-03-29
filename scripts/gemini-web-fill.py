#!/usr/bin/env python3
"""
Gemini Web UI automation — fill weak classes with pixel art sprites.
Uses Chrome with existing profile (logged in). No API key needed.

Usage: python3 scripts/gemini-web-fill.py [data_v3_32]
"""

import os
import sys
import time
import base64
import json
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = [a for a in sys.argv[1:] if a.startswith("--")]
DST = args[0] if args else "data_v3_32"
TARGET = 200
MANUAL_MODE = "--manual" in flags
DELAY = 5  # seconds between prompts
CHROME_PROFILE = os.path.expanduser("~/Library/Application Support/Google/Chrome")
CHROMEDRIVER = "/opt/homebrew/bin/chromedriver"

CLASS_DESCRIPTIONS = {
    "bush_flower": "bush with colorful flowers and green leaves",
    "cat_space": "space cat in a tiny helmet, floating, sci-fi",
    "cat_warrior": "cat warrior with sword and armor, medieval fighter",
    "cat_fat": "fat round chubby cat sitting down, cute",
    "cat_wizard": "cat wearing a wizard hat with a magic staff",
    "crop": "farm crop plant like wheat or corn or tomato",
    "cyborg": "cyborg with half human half machine face, glowing eye",
    "door_stair": "dungeon door or stone staircase, RPG tile",
    "dwarf": "dwarf warrior with beard and axe, mining helmet",
    "elf": "elf archer with pointed ears and bow, forest theme",
    "farm_animal": "farm animal like cow or pig or chicken or sheep",
    "fish": "colorful tropical fish, aquatic creature",
    "fx_ambient": "ambient sparkle or dust mote or firefly effect",
    "fx_combat": "combat slash effect or impact explosion",
    "goblin": "goblin with green skin and dagger, sneaky and small",
    "ground_natural": "natural ground tile of grass or dirt or sand",
    "hero_cleric": "cleric healer in robes with holy staff",
    "horse": "horse in side view, war horse or farm horse",
    "mech": "bipedal mech robot, sci-fi walker, armed",
    "mount": "rideable fantasy mount creature with saddle",
    "mushroom": "fantasy mushroom with red cap, glowing",
    "reptile": "reptile like lizard or snake or turtle",
    "slime": "bouncy slime blob, translucent, RPG enemy",
    "spear_polearm": "spear or polearm weapon, long shaft with blade",
    "terrain_scifi": "sci-fi metal floor tile or neon panel",
    "tree_conifer": "conifer pine tree, evergreen, pointed top",
    "tree_exotic": "exotic tropical tree like palm or baobab",
    "wall": "stone brick wall segment, dungeon or castle",
    "wild_animal": "wild forest animal like deer or fox or rabbit",
    "wolf": "wolf walking or howling, gray fur",
    "zombie": "zombie with green rotting skin, undead RPG enemy",
    "dog": "dog pet, varied breed, sitting or standing",
    # Empty classes
    "alien": "grey alien with big eyes, sci-fi",
    "bat": "flying bat, cave bat or vampire bat",
    "cat_mech": "mechanical robot cat, steampunk or cyborg",
    "dinosaur": "dinosaur like t-rex or raptor or triceratops",
    "furniture_scifi": "sci-fi furniture like control panel or stasis pod",
    "golem": "stone golem, hulking, rocky, glowing eyes",
    "insect": "insect like beetle or ant or butterfly",
    "mythical_beast": "mythical beast like griffin or chimera or manticore",
    "spider": "spider, large, hairy, multiple eyes",
    "animal": "generic animal like raccoon or hedgehog or squirrel",
    "vehicle": "generic vehicle like cart or wagon or mine cart",
    "water_lava": "water tile or lava tile, blue or red flowing",
}


def get_weak_classes(dst):
    """Find classes under TARGET samples."""
    weak = []
    dst_path = Path(dst)
    for class_dir in sorted(dst_path.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith(("_", ".")):
            continue
        count = len(list(class_dir.glob("*.png")))
        if count < TARGET:
            weak.append((class_dir.name, count, TARGET - count))
    return weak


def setup_chrome():
    """Launch Chrome with existing profile."""
    opts = Options()
    opts.binary_location = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    # Use a copy of the profile to avoid lock conflicts
    opts.add_argument(f"--user-data-dir={CHROME_PROFILE}")
    opts.add_argument("--profile-directory=Default")
    opts.add_argument("--no-first-run")
    opts.add_argument("--no-default-browser-check")
    # Don't use headless — Gemini may block it
    opts.add_argument("--window-size=1280,900")

    service = Service(CHROMEDRIVER)
    driver = webdriver.Chrome(service=service, options=opts)
    return driver


def extract_images_from_page(driver):
    """Extract all generated images from Gemini response."""
    images = []
    # Gemini renders images as <img> tags inside response containers
    # Try multiple selectors
    selectors = [
        "img[src^='data:image']",
        "img[src^='blob:']",
        "response-image img",
        ".response-container img",
        "message-content img",
        "model-response img",
    ]

    for sel in selectors:
        try:
            elements = driver.find_elements(By.CSS_SELECTOR, sel)
            for el in elements:
                src = el.get_attribute("src")
                if src and src.startswith("data:image"):
                    # data URI
                    _, encoded = src.split(",", 1)
                    images.append(base64.b64decode(encoded))
                elif src and src.startswith("blob:"):
                    # Blob URL — need to fetch via JS
                    b64 = driver.execute_script("""
                        return await fetch(arguments[0])
                            .then(r => r.blob())
                            .then(b => new Promise((res) => {
                                const reader = new FileReader();
                                reader.onload = () => res(reader.result.split(',')[1]);
                                reader.readAsDataURL(b);
                            }));
                    """, src)
                    if b64:
                        images.append(base64.b64decode(b64))
        except Exception:
            pass

    # Fallback: grab all canvas elements
    if not images:
        try:
            canvases = driver.find_elements(By.TAG_NAME, "canvas")
            for canvas in canvases:
                data_url = driver.execute_script(
                    "return arguments[0].toDataURL('image/png');", canvas
                )
                if data_url:
                    _, encoded = data_url.split(",", 1)
                    images.append(base64.b64decode(encoded))
        except Exception:
            pass

    return images


def resize_to_32(img_bytes, outpath):
    """Resize image to 32x32 using nearest neighbor, save as PNG."""
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(img_bytes))
        if img.size != (32, 32):
            img = img.resize((32, 32), Image.NEAREST)
        img.save(outpath, "PNG")
        return True
    except Exception as e:
        # Fallback: save raw and hope for the best
        with open(outpath, "wb") as f:
            f.write(img_bytes)
        return True


def generate_sprite(driver, class_name, description):
    """Navigate to Gemini, type prompt, extract image (grid of 30 sprites)."""
    prompt = (
        f"Generate a 6 by 5 grid of 32x32 pixel art sprites of {description}. "
        f"Each sprite different. 8-bit retro game style, clean single-pixel edges, "
        f"solid black background, game asset quality. 30 unique sprites in a grid."
    )

    # Navigate fresh each time to avoid context buildup
    driver.get("https://gemini.google.com/app")
    time.sleep(3)

    # Find the input area
    input_sel = [
        "div[contenteditable='true']",
        "rich-textarea div[contenteditable]",
        ".ql-editor",
        "textarea",
        "[aria-label*='prompt']",
        "[aria-label*='Enter']",
    ]

    input_el = None
    for sel in input_sel:
        try:
            input_el = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, sel))
            )
            if input_el:
                break
        except Exception:
            continue

    if not input_el:
        print(f"  SKIP: could not find input element")
        return None

    # Type the prompt
    input_el.click()
    time.sleep(0.5)
    input_el.send_keys(prompt)
    time.sleep(0.5)

    # Submit — Enter or find send button
    try:
        send_btn = driver.find_element(By.CSS_SELECTOR, "button[aria-label*='Send'], button[aria-label*='submit'], .send-button")
        send_btn.click()
    except Exception:
        input_el.send_keys(Keys.RETURN)

    # Wait for response (image generation takes 10-30s)
    print(f"  waiting for generation...")
    time.sleep(15)

    # Poll for images up to 60 seconds
    for _ in range(10):
        # Check if response is still loading
        try:
            loading = driver.find_elements(By.CSS_SELECTOR, ".loading, .generating, [aria-busy='true']")
            if loading:
                time.sleep(5)
                continue
        except Exception:
            pass

        images = extract_images_from_page(driver)
        if images:
            return images[-1]  # Return the last (most recent) image
        time.sleep(5)

    print(f"  SKIP: no image found after 60s")
    return None


def main():
    weak = get_weak_classes(DST)
    if not weak:
        print("all classes at target — nothing to fill")
        return

    print(f"weak classes: {len(weak)}")
    total_needed = sum(n for _, _, n in weak)
    print(f"total sprites needed: {total_needed}")
    print()

    # Check if we should use automated browser or manual mode
    if MANUAL_MODE:
        print("MANUAL MODE: printing prompts only")
        for cls, current, needed in weak:
            desc = CLASS_DESCRIPTIONS.get(cls, f"{cls}, fantasy RPG game asset")
            print(f"\n--- {cls} (need {needed}) ---")
            print(f"Generate a single 32x32 pixel art sprite of a {desc}. "
                  f"8-bit retro game style, clean single-pixel edges, solid black background, "
                  f"centered in frame, no text. Show only the sprite.")
        return

    print("launching Chrome with existing profile...")
    driver = setup_chrome()

    try:
        total_generated = 0
        for cls, current, needed in weak:
            desc = CLASS_DESCRIPTIONS.get(cls, f"{cls}, fantasy RPG game asset")
            class_dir = Path(DST) / cls
            class_dir.mkdir(exist_ok=True)

            print(f"\n=== {cls}: {current} → {TARGET} (need {needed}) ===")

            generated = 0
            idx = current

            while generated < needed:
                idx += 1
                outpath = class_dir / f"gemini_web_{idx:04d}.png"

                img_data = generate_sprite(driver, cls, desc)
                if img_data:
                    resize_to_32(img_data, str(outpath))
                    generated += 1
                    total_generated += 1
                    print(f"  [{generated}/{needed}] saved {outpath.name}")
                else:
                    print(f"  failed — retrying in 10s")
                    time.sleep(10)
                    # Try once more, then skip this sprite
                    img_data = generate_sprite(driver, cls, desc)
                    if img_data:
                        resize_to_32(img_data, str(outpath))
                        generated += 1
                        total_generated += 1
                        print(f"  [{generated}/{needed}] saved {outpath.name} (retry)")
                    else:
                        print(f"  skipping sprite {idx}")

                time.sleep(DELAY)

            new_count = len(list(class_dir.glob("*.png")))
            print(f"  done: {cls} = {new_count} samples")

        print(f"\ntotal generated: {total_generated}")

    finally:
        driver.quit()

    # Update manifest
    print("updating manifest...")
    manifest = Path(DST) / "MANIFEST.md"
    lines = ["# data_v3_32 Manifest (post-fill)\n", "\n",
             "Rebalanced + Gemini web-filled dataset.\n", "\n",
             "| Class | Count |\n", "|-------|-------|\n"]
    grand = 0
    for d in sorted(Path(DST).iterdir()):
        if d.is_dir() and not d.name.startswith(("_", ".")):
            c = len(list(d.glob("*.png")))
            if c > 0:
                lines.append(f"| {d.name} | {c} |\n")
                grand += c
    lines.append(f"\nTotal: {grand}\n")
    manifest.write_text("".join(lines))
    print(f"manifest updated. total: {grand}")


if __name__ == "__main__":
    main()
