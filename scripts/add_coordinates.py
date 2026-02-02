"""Add GPS coordinates to cities database from original source."""

import csv
import sqlite3
import urllib.request
from pathlib import Path

from tqdm import tqdm


DB_PATH = Path(__file__).parent.parent / "data" / "cities.db"
SOURCE_URL = "https://raw.githubusercontent.com/dr5hn/countries-states-cities-database/master/csv/cities.csv"


def add_coordinates():
    """Download source CSV and add lat/lon to database."""
    conn = sqlite3.connect(DB_PATH)
    
    # Add columns if they don't exist
    try:
        conn.execute("ALTER TABLE cities ADD COLUMN latitude REAL")
        conn.execute("ALTER TABLE cities ADD COLUMN longitude REAL")
        conn.commit()
        print("Added latitude/longitude columns")
    except sqlite3.OperationalError:
        print("Latitude/longitude columns already exist")
    
    # Download source
    print(f"Downloading source data from {SOURCE_URL}...")
    with urllib.request.urlopen(SOURCE_URL) as response:
        content = response.read().decode("utf-8")
    
    reader = csv.DictReader(content.splitlines())
    
    # Build lookup: (city, region, country) -> (lat, lon)
    coords = {}
    for row in reader:
        key = (row["name"], row.get("state_name", ""), row["country_name"])
        try:
            lat = float(row["latitude"])
            lon = float(row["longitude"])
            coords[key] = (lat, lon)
        except (ValueError, KeyError):
            continue
    
    print(f"Loaded {len(coords)} city coordinates")
    
    # Update database
    cursor = conn.cursor()
    cities = cursor.execute(
        "SELECT id, city, region, country FROM cities WHERE latitude IS NULL"
    ).fetchall()
    
    updated = 0
    for city_id, city, region, country in tqdm(cities, desc="Updating coordinates"):
        key = (city, region or "", country)
        if key in coords:
            lat, lon = coords[key]
            cursor.execute(
                "UPDATE cities SET latitude = ?, longitude = ? WHERE id = ?",
                (lat, lon, city_id)
            )
            updated += 1
    
    conn.commit()
    
    # Stats
    total = cursor.execute("SELECT COUNT(*) FROM cities").fetchone()[0]
    with_coords = cursor.execute(
        "SELECT COUNT(*) FROM cities WHERE latitude IS NOT NULL"
    ).fetchone()[0]
    
    print(f"Updated {updated} cities with coordinates")
    print(f"Total: {with_coords}/{total} cities have coordinates")
    
    conn.close()


if __name__ == "__main__":
    add_coordinates()
