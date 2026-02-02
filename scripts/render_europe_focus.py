"""Render high-resolution Europe-focused heatmap.

Bounds: 30W to 60E longitude, 30N to 75N latitude
Target: ~4M pixels (2828 x 1414)
"""

import numpy as np
import subprocess
from pathlib import Path
from PIL import Image
import matplotlib.colors as mcolors
from tqdm import tqdm

from europe_or_not.scorer import EuropeScorer, EARTH_RADIUS_KM


# Region bounds
LON_MIN, LON_MAX = -30, 60  # 30W to 60E
LAT_MIN, LAT_MAX = 30, 75   # 30N to 75N

# Target ~4M pixels with 2:1 aspect ratio (90° lon x 45° lat)
WIDTH = 2828
HEIGHT = 1414

OUTPUT_PATH = Path(__file__).parent.parent / "europe_focus_heatmap.png"
SVG_PATH = Path(__file__).parent.parent / "equi.svg"
RASTER_PATH = Path(__file__).parent.parent / "equi_hires.png"


def create_heatmap_colormap():
    """Create a colormap from blue (not Europe) to red (Europe)."""
    colors = [
        (0.0, 0.2, 0.6),    # Dark blue (0.0 - definitely not Europe)
        (0.0, 0.5, 0.8),    # Blue
        (0.0, 0.8, 0.8),    # Cyan
        (0.2, 0.8, 0.2),    # Green
        (0.8, 0.8, 0.0),    # Yellow
        (1.0, 0.5, 0.0),    # Orange
        (0.9, 0.1, 0.1),    # Red (1.0 - definitely Europe)
    ]
    return mcolors.LinearSegmentedColormap.from_list("europe", colors)


def haversine_vectorized(lat1: float, lon1: float, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Vectorized haversine distance in km."""
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lats_rad = np.radians(lats)
    lons_rad = np.radians(lons)

    dlat = lats_rad - lat1_rad
    dlon = lons_rad - lon1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lats_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return EARTH_RADIUS_KM * c


class FastEuropeScorer:
    """Optimized scorer using vectorized operations."""

    def __init__(self, scorer: EuropeScorer):
        self.bandwidth = scorer.kernel_bandwidth_km
        self.min_weight = scorer.min_weight
        self.city_lats = np.array([c.lat for c in scorer.cities])
        self.city_lons = np.array([c.lon for c in scorer.cities])
        self.city_scores = np.array([c.europe_score for c in scorer.cities])
        print(f"FastEuropeScorer initialized with {len(self.city_lats)} cities")

    def score(self, lat: float, lon: float) -> float:
        distances = haversine_vectorized(lat, lon, self.city_lats, self.city_lons)
        weights = np.exp(-0.5 * (distances / self.bandwidth) ** 2)
        mask = weights > self.min_weight
        if not mask.any():
            return 0.5
        return np.sum(weights[mask] * self.city_scores[mask]) / np.sum(weights[mask])


def render_svg_to_raster(svg_path: Path, output_path: Path, width: int, height: int) -> None:
    """Render SVG to high-res raster using ImageMagick."""
    # Calculate full world dimensions to get proper crop
    # Full world is 360° x 180°, we want a portion
    
    # For equirectangular, the full SVG covers -180 to 180 lon, 90 to -90 lat
    # We want -30 to 60 lon (90°), 30 to 75 lat (45°)
    
    # Calculate what resolution we need for the full world
    # to get our desired crop resolution
    lon_range = LON_MAX - LON_MIN  # 90°
    lat_range = LAT_MAX - LAT_MIN  # 45°
    
    full_width = int(WIDTH * 360 / lon_range)
    full_height = int(HEIGHT * 180 / lat_range)
    
    print(f"Rendering SVG at {full_width}x{full_height} for crop...")
    
    # Render full SVG
    subprocess.run([
        "convert", "-background", "white", "-density", "300",
        str(svg_path), "-resize", f"{full_width}x{full_height}!",
        str(output_path)
    ], check=True)
    
    print(f"Saved full raster to {output_path}")


def crop_to_region(img: Image.Image, full_width: int, full_height: int) -> Image.Image:
    """Crop image to the Europe focus region."""
    # Calculate pixel coordinates for our bounds
    # x: -180 to 180 maps to 0 to full_width
    # y: 90 to -90 maps to 0 to full_height
    
    x_min = int((LON_MIN + 180) / 360 * full_width)
    x_max = int((LON_MAX + 180) / 360 * full_width)
    y_min = int((90 - LAT_MAX) / 180 * full_height)
    y_max = int((90 - LAT_MIN) / 180 * full_height)
    
    print(f"Cropping to region: x=[{x_min}:{x_max}], y=[{y_min}:{y_max}]")
    return img.crop((x_min, y_min, x_max, y_max))


def render_europe_focus(
    scorer: EuropeScorer,
    output_path: Path = OUTPUT_PATH,
) -> Path:
    """Render focused Europe heatmap."""
    fast_scorer = FastEuropeScorer(scorer)
    
    # Check if we need to render the SVG
    if not RASTER_PATH.exists():
        print("Rendering SVG to raster...")
        render_svg_to_raster(SVG_PATH, RASTER_PATH, WIDTH, HEIGHT)
    
    # Load and crop the raster
    print("Loading base map...")
    base_img = Image.open(RASTER_PATH).convert("RGBA")
    full_width, full_height = base_img.size
    print(f"Full raster size: {full_width}x{full_height}")
    
    # Crop to region
    cropped = crop_to_region(base_img, full_width, full_height)
    width, height = cropped.size
    print(f"Cropped size: {width}x{height} ({width * height:,} pixels)")
    
    # Get grayscale values for land detection
    # The SVG renders as grayscale: 0 = black (land), 255 = white (ocean)
    gray = np.array(cropped.convert('L'))
    
    # Land is dark (< 128), ocean is light (>= 128)
    is_land = gray < 128
    
    land_y, land_x = np.where(is_land)
    print(f"Found {len(land_x):,} land pixels")
    
    # Convert pixel coords to lat/lon
    lons = LON_MIN + (land_x / width) * (LON_MAX - LON_MIN)
    lats = LAT_MAX - (land_y / height) * (LAT_MAX - LAT_MIN)
    
    # Create output array
    output = np.ones((height, width, 3), dtype=np.float32)  # White background
    cmap = create_heatmap_colormap()
    
    # Score all land pixels
    print("Computing Europe scores...")
    scores = np.zeros(len(lats))
    for i in tqdm(range(len(lats)), desc="Scoring pixels"):
        scores[i] = fast_scorer.score(lats[i], lons[i])
    
    # Apply colormap
    print("Applying colormap...")
    colors = cmap(scores)[:, :3]
    output[land_y, land_x] = colors
    
    # Save
    output_img = Image.fromarray((output * 255).astype(np.uint8))
    output_img.save(output_path, quality=95)
    print(f"Saved heatmap to {output_path}")
    
    return output_path


def main():
    print("Initializing scorer...")
    scorer = EuropeScorer(kernel_bandwidth_km=500.0)
    
    print(f"\nRendering Europe focus heatmap...")
    print(f"Region: {LON_MIN}° to {LON_MAX}° lon, {LAT_MIN}° to {LAT_MAX}° lat")
    print(f"Target size: {WIDTH}x{HEIGHT} ({WIDTH * HEIGHT:,} pixels)")
    
    output = render_europe_focus(scorer)
    print(f"\nDone! Output: {output}")


if __name__ == "__main__":
    main()
