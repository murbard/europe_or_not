"""Download and filter world cities dataset to ~3000 cities with good global distribution."""

import csv
import urllib.request
from pathlib import Path


def download_and_filter_cities(output_path: Path, target_count: int = 3000) -> None:
    """Download cities CSV and filter to prominent cities with global distribution.
    
    Strategy:
    1. Download the full dataset
    2. Sort by population (descending) 
    3. Take top cities to get target count
    4. Save with city, country, and region columns
    """
    url = "https://raw.githubusercontent.com/dr5hn/countries-states-cities-database/master/csv/cities.csv"
    
    print(f"Downloading cities from {url}...")
    with urllib.request.urlopen(url) as response:
        content = response.read().decode("utf-8")
    
    # Parse CSV
    reader = csv.DictReader(content.splitlines())
    cities = []
    
    for row in reader:
        # Skip entries without population data
        pop_str = row.get("population", "").strip()
        if not pop_str:
            continue
        try:
            population = int(pop_str)
        except ValueError:
            continue
        
        cities.append({
            "city": row["name"],
            "region": row.get("state_name", ""),
            "country": row["country_name"],
            "population": population,
        })
    
    print(f"Found {len(cities)} cities with population data")
    
    # Sort by population descending and take top N
    cities.sort(key=lambda x: x["population"], reverse=True)
    filtered = cities[:target_count]
    
    print(f"Selected top {len(filtered)} cities by population")
    
    # Get country distribution stats
    countries = {}
    for c in filtered:
        countries[c["country"]] = countries.get(c["country"], 0) + 1
    print(f"Cities span {len(countries)} countries")
    
    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["city", "region", "country"])
        writer.writeheader()
        for c in filtered:
            writer.writerow({
                "city": c["city"],
                "region": c["region"],
                "country": c["country"],
            })
    
    print(f"Wrote {len(filtered)} cities to {output_path}")


if __name__ == "__main__":
    output = Path(__file__).parent.parent / "data" / "cities.csv"
    download_and_filter_cities(output, target_count=3000)
