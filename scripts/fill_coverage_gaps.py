"""Find coverage gaps and add real cities via reverse geocoding."""

import numpy as np
from PIL import Image
from pathlib import Path
import sqlite3
import requests
import time

EARTH_RADIUS_KM = 6371.0

# Region bounds
LON_MIN, LON_MAX = -30, 60
LAT_MIN, LAT_MAX = 30, 75

MAX_DISTANCE_KM = 100

OUTPUT_DIR = Path(__file__).parent.parent
RASTER_PATH = OUTPUT_DIR / "equi_hires.png"
DB_PATH = OUTPUT_DIR / "data" / "cities.db"


def haversine_vectorized(lat1, lon1, lats, lons):
    """Vectorized haversine distance in km."""
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lats_rad, lons_rad = np.radians(lats), np.radians(lons)
    dlat, dlon = lats_rad - lat1_rad, lons_rad - lon1_rad
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lats_rad) * np.sin(dlon / 2) ** 2
    return EARTH_RADIUS_KM * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def load_cities():
    """Load all cities with coordinates."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT latitude, longitude FROM cities WHERE latitude IS NOT NULL")
    cities = [(lat, lon) for lat, lon in cursor]
    conn.close()
    return np.array(cities) if cities else np.array([]).reshape(0, 2)


def get_land_points():
    """Get sampled land points from raster."""
    img = Image.open(RASTER_PATH).convert("RGBA")
    full_w, full_h = img.size
    
    # Crop to region
    x_min = int((LON_MIN + 180) / 360 * full_w)
    x_max = int((LON_MAX + 180) / 360 * full_w)
    y_min = int((90 - LAT_MAX) / 180 * full_h)
    y_max = int((90 - LAT_MIN) / 180 * full_h)
    cropped = img.crop((x_min, y_min, x_max, y_max))
    width, height = cropped.size
    
    gray = np.array(cropped.convert('L'))
    is_land = gray < 128
    
    # Sample every 5 pixels for better resolution
    step = 5
    land_y, land_x = np.where(is_land[::step, ::step])
    land_y, land_x = land_y * step, land_x * step
    
    lons = LON_MIN + (land_x / width) * (LON_MAX - LON_MIN)
    lats = LAT_MAX - (land_y / height) * (LAT_MAX - LAT_MIN)
    
    return np.column_stack([lats, lons])


def find_worst_gap(land_points, city_coords, temp_points):
    """Find the land point furthest from any city or temp point."""
    all_points = np.vstack([city_coords, temp_points]) if len(temp_points) > 0 else city_coords
    
    max_dist = 0
    worst_point = None
    
    for lat, lon in land_points:
        distances = haversine_vectorized(lat, lon, all_points[:, 0], all_points[:, 1])
        min_dist = np.min(distances)
        if min_dist > max_dist:
            max_dist = min_dist
            worst_point = (lat, lon)
    
    return worst_point, max_dist


def reverse_geocode(lat, lon):
    """Use Nominatim to find nearest city."""
    url = f"https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": lat,
        "lon": lon,
        "format": "json",
        "zoom": 10,  # City level
        "addressdetails": 1
    }
    headers = {"User-Agent": "europe_or_not/1.0"}
    
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        address = data.get("address", {})
        
        # Try to get city name
        city = (address.get("city") or address.get("town") or 
                address.get("village") or address.get("municipality") or
                address.get("county") or address.get("state"))
        country = address.get("country", "Unknown")
        
        if city:
            return city, country, float(data.get("lat", lat)), float(data.get("lon", lon))
    except Exception as e:
        print(f"  Geocode error: {e}")
    
    return None, None, lat, lon


def city_exists(city, country):
    """Check if city already exists in database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT id FROM cities WHERE city = ? AND country = ?",
        (city, country)
    )
    exists = cursor.fetchone() is not None
    conn.close()
    return exists


def add_city(city, country, lat, lon, label=None):
    """Add a city to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO cities (city, region, country, latitude, longitude, label) VALUES (?, ?, ?, ?, ?, ?)",
        (city, "", country, lat, lon, label)
    )
    conn.commit()
    conn.close()


def main():
    land_points = get_land_points()
    print(f"Loaded {len(land_points)} land points")
    
    temp_points = []  # Temporary points for finding different gaps
    cities_added = 0
    max_iterations = 2000
    
    for iteration in range(max_iterations):
        cities = load_cities()
        print(f"\nIteration {iteration + 1}: {len(cities)} cities in DB")
        
        # Find worst gap
        worst_point, max_dist = find_worst_gap(land_points, cities, np.array(temp_points) if temp_points else np.array([]).reshape(0, 2))
        
        lat, lon = worst_point
        print(f"Worst gap: {max_dist:.1f}km at ({lat:.2f}, {lon:.2f})")
        
        # Rate limit for Nominatim
        time.sleep(1.1)
        
        # Reverse geocode to find nearest city
        city, country, city_lat, city_lon = reverse_geocode(lat, lon)
        
        if city is None:
            print(f"  No city found, adding temp point")
            temp_points.append([lat, lon])
            continue
        
        print(f"  Found: {city}, {country} at ({city_lat:.2f}, {city_lon:.2f})")
        
        if city_exists(city, country):
            print(f"  Already in DB, adding temp point")
            temp_points.append([lat, lon])
        else:
            add_city(city, country, city_lat, city_lon)
            cities_added += 1
            print(f"  Added to DB! ({cities_added} total added)")
    
    print(f"\n=== Summary ===")
    print(f"Added {cities_added} new cities")
    print(f"Created {len(temp_points)} temp points (for gap finding only)")
    print(f"Run collect_embeddings.py to get embeddings for new cities")


if __name__ == "__main__":
    main()
