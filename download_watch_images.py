#!/usr/bin/env python3
"""
Download Watch Images from main_image URLs in final_scrape.csv
============================================================

This script downloads all images from the 'main_image' column in the CSV
and saves them to the local images directory, using a safe filename
based on brand and model. Skips images that already exist.
"""

import os
import pandas as pd
import requests
import time
import re
from urllib.parse import urlparse

# Config
CSV_PATH = "production_scrape_20250601_175426/data/final_scrape.csv"
IMAGES_DIR = "production_scrape_20250601_175426/images"
SLEEP_BETWEEN = 0.1  # seconds between downloads
TIMEOUT = 15         # seconds for HTTP requests

os.makedirs(IMAGES_DIR, exist_ok=True)

def safe_filename(brand, model):
    """Create a safe filename for the image."""
    name = f"{brand}_{model}_main.jpg"
    # Remove/replace unsafe characters
    name = re.sub(r"[^A-Za-z0-9_\-\.]+", "_", name)
    return name

def download_image(url, path):
    try:
        response = requests.get(url, timeout=TIMEOUT)
        if response.status_code == 200:
            with open(path, 'wb') as f:
                f.write(response.content)
            return True
        else:
            print(f"❌ Failed to download {url} (status {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return False

def main():
    print(f"📖 Reading CSV: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)
    print(f"✅ Loaded {len(df)} watches")
    
    downloaded = 0
    skipped = 0
    failed = 0
    
    for idx, row in df.iterrows():
        brand = str(row.get('brand', 'Unknown')).strip()
        model = str(row.get('model', 'Unknown')).strip()
        url = row.get('main_image', '').strip()
        if not url or not url.startswith('http'):
            print(f"⚠️  No valid image URL for {brand} {model}")
            failed += 1
            continue
        
        filename = safe_filename(brand, model)
        out_path = os.path.join(IMAGES_DIR, filename)
        
        if os.path.exists(out_path):
            skipped += 1
            continue
        
        print(f"⬇️  Downloading: {brand} {model} → {filename}")
        if download_image(url, out_path):
            downloaded += 1
        else:
            failed += 1
        time.sleep(SLEEP_BETWEEN)
    
    print(f"\n✅ Done! Downloaded: {downloaded}, Skipped: {skipped}, Failed: {failed}")

if __name__ == "__main__":
    main() 