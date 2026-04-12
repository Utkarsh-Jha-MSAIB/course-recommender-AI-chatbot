import os
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# =========================================================
# CONFIG
# =========================================================
load_dotenv()

INPUT_FILE = os.getenv("INPUT_FILE", "./data/courses.xlsx")
SHEET_NAME = os.getenv("SHEET_NAME", "Catalog_Python")
OUTPUT_FILE = os.getenv("IMAGE_ENRICHED_OUTPUT_FILE", "./data/courses_with_images.xlsx")

REQUEST_TIMEOUT = int(os.getenv("IMAGE_FETCH_TIMEOUT", "12"))
REQUEST_DELAY_SECONDS = float(os.getenv("IMAGE_FETCH_DELAY_SECONDS", "1.0"))
OVERWRITE_EXISTING = os.getenv("IMAGE_OVERWRITE_EXISTING", "false").lower() == "true"
MAX_ROWS = int(os.getenv("IMAGE_ENRICH_MAX_ROWS", "0"))  # 0 = all

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "en-US,en;q=0.9",
}

# =========================================================
# HELPERS
# =========================================================
def clean_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def is_likely_image_url(url: str) -> bool:
    url = url.lower()
    image_extensions = [".jpg", ".jpeg", ".png", ".webp", ".gif", ".avif"]
    return any(ext in url for ext in image_extensions)


def fetch_html(url: str) -> Optional[str]:
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return response.text
    except Exception as exc:
        print(f"[WARN] Failed to fetch {url} -> {exc}")
        return None


def get_meta_content(soup: BeautifulSoup, attr_name: str, attr_value: str) -> str:
    tag = soup.find("meta", attrs={attr_name: attr_value})
    if tag and tag.get("content"):
        return tag["content"].strip()
    return ""


def extract_image_from_meta(soup: BeautifulSoup, page_url: str) -> str:
    candidates = [
        get_meta_content(soup, "property", "og:image"),
        get_meta_content(soup, "property", "og:image:url"),
        get_meta_content(soup, "name", "twitter:image"),
        get_meta_content(soup, "name", "twitter:image:src"),
    ]

    for candidate in candidates:
        if candidate:
            return urljoin(page_url, candidate)

    return ""


def extract_image_from_img_tags(soup: BeautifulSoup, page_url: str) -> str:
    selectors = [
        {"attrs": {"data-testid": "hero-image"}},
        {"attrs": {"class": lambda x: x and "hero" in x.lower()}},
        {"attrs": {"class": lambda x: x and "banner" in x.lower()}},
    ]

    for selector in selectors:
        try:
            tag = soup.find("img", **selector)
            if tag:
                src = tag.get("src") or tag.get("data-src") or tag.get("srcset")
                if src:
                    src = src.split(",")[0].strip().split(" ")[0].strip()
                    return urljoin(page_url, src)
        except Exception:
            pass

    for tag in soup.find_all("img"):
        src = tag.get("src") or tag.get("data-src")
        if not src:
            continue

        full_src = urljoin(page_url, src.strip())

        alt_text = (tag.get("alt") or "").lower()
        class_text = " ".join(tag.get("class", [])).lower()

        if (
            "course" in alt_text
            or "course" in class_text
            or "hero" in class_text
            or "banner" in class_text
            or is_likely_image_url(full_src)
        ):
            return full_src

    return ""


def extract_course_image_url(page_url: str) -> str:
    html = fetch_html(page_url)
    if not html:
        return ""

    try:
        soup = BeautifulSoup(html, "html.parser")
    except Exception as exc:
        print(f"[WARN] Failed to parse HTML for {page_url} -> {exc}")
        return ""

    image_url = extract_image_from_meta(soup, page_url)
    if image_url:
        return image_url

    image_url = extract_image_from_img_tags(soup, page_url)
    if image_url:
        return image_url

    return ""


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Reading: {input_path}")
    df = pd.read_excel(input_path, sheet_name=SHEET_NAME)

    if "Course URL" not in df.columns:
        raise ValueError("Missing required column: 'Course URL'")

    if "Course Image URL" not in df.columns:
        df["Course Image URL"] = ""

    total_rows = len(df)
    print(f"Loaded rows: {total_rows}")

    processed = 0
    filled = 0
    skipped_existing = 0
    failed = 0

    row_indices = df.index.tolist()
    if MAX_ROWS > 0:
        row_indices = row_indices[:MAX_ROWS]

    for idx in row_indices:
        course_name = clean_text(df.at[idx, "Course Name"]) if "Course Name" in df.columns else f"Row {idx}"
        course_url = clean_text(df.at[idx, "Course URL"])
        existing_image_url = clean_text(df.at[idx, "Course Image URL"])

        if not course_url:
            print(f"[SKIP] No course URL for: {course_name}")
            failed += 1
            continue

        if existing_image_url and not OVERWRITE_EXISTING:
            print(f"[SKIP] Already has image URL: {course_name}")
            skipped_existing += 1
            continue

        print(f"[FETCH] {course_name}")
        image_url = extract_course_image_url(course_url)

        if image_url:
            df.at[idx, "Course Image URL"] = image_url
            filled += 1
            print(f"       -> {image_url}")
        else:
            print("       -> No image found")
            failed += 1

        processed += 1
        time.sleep(REQUEST_DELAY_SECONDS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)

    print("\nDone.")
    print(f"Processed: {processed}")
    print(f"Filled: {filled}")
    print(f"Skipped existing: {skipped_existing}")
    print(f"Failed / not found: {failed}")
    print(f"Saved enriched file to: {output_path}")


if __name__ == "__main__":
    main()