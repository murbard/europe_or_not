"""Optimized render of equirectangular heatmap using vectorized operations."""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.colors as mcolors
from tqdm import tqdm

from europe_or_not.scorer import EuropeScorer, haversine_distance, EARTH_RADIUS_KM


BASE_MAP = Path(__file__).parent.parent / "equi.png"
OUTPUT_PATH = Path(__file__).parent.parent / "europe_heatmap.png"


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
    """Vectorized haversine distance calculation.

    Args:
        lat1, lon1: Single query point in degrees
        lats, lons: Arrays of city coordinates in degrees

    Returns:
        Array of distances in km
    """
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
        """Initialize from an existing scorer."""
        self.bandwidth = scorer.kernel_bandwidth_km
        self.min_weight = scorer.min_weight

        # Pre-compute numpy arrays for vectorized operations
        self.city_lats = np.array([c.lat for c in scorer.cities])
        self.city_lons = np.array([c.lon for c in scorer.cities])
        self.city_scores = np.array([c.europe_score for c in scorer.cities])

        print(f"FastEuropeScorer initialized with {len(self.city_lats)} cities")

    def score(self, lat: float, lon: float) -> float:
        """Compute Europe score using vectorized operations."""
        # Vectorized distance calculation
        distances = haversine_vectorized(lat, lon, self.city_lats, self.city_lons)

        # Vectorized Gaussian kernel
        weights = np.exp(-0.5 * (distances / self.bandwidth) ** 2)

        # Apply minimum weight threshold
        mask = weights > self.min_weight
        if not mask.any():
            return 0.5

        valid_weights = weights[mask]
        valid_scores = self.city_scores[mask]

        return np.sum(valid_weights * valid_scores) / np.sum(valid_weights)

    def score_batch(self, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
        """Score multiple coordinates at once.

        This is even faster for large batches by avoiding Python loops.
        """
        n_points = len(lats)
        scores = np.zeros(n_points)

        # Process in chunks to balance memory and speed
        chunk_size = 1000
        for i in range(0, n_points, chunk_size):
            end_i = min(i + chunk_size, n_points)
            for j, (lat, lon) in enumerate(zip(lats[i:end_i], lons[i:end_i])):
                scores[i + j] = self.score(lat, lon)

        return scores


def render_heatmap_fast(
    scorer: EuropeScorer,
    base_map_path: Path = BASE_MAP,
    output_path: Path = OUTPUT_PATH,
    sample_step: int = 1,
) -> Path:
    """Render Europe score heatmap using optimized vectorized operations."""
    # Create fast scorer
    fast_scorer = FastEuropeScorer(scorer)

    # Load base map
    base_img = Image.open(base_map_path).convert("RGBA")
    width, height = base_img.size
    print(f"Base map size: {width}x{height}")

    # Get alpha channel as numpy array for fast land detection
    alpha = np.array(base_img)[:, :, 3]

    # Create coordinate grids
    x_coords = np.arange(0, width, sample_step)
    y_coords = np.arange(0, height, sample_step)

    # Create output array (white background)
    output = np.ones((height, width, 3), dtype=np.float32)

    # Get colormap
    cmap = create_heatmap_colormap()

    # Find land pixels using numpy (much faster)
    land_mask_sampled = alpha[::sample_step, ::sample_step] > 0
    land_y, land_x = np.where(land_mask_sampled)
    land_x = land_x * sample_step
    land_y = land_y * sample_step

    print(f"Found {len(land_x)} land pixels to process")

    # Convert pixel coordinates to lat/lon
    lons = (land_x / width) * 360 - 180
    lats = 90 - (land_y / height) * 180

    # Score all land pixels
    print("Computing Europe scores (vectorized)...")
    scores = np.zeros(len(lats))
    for i in tqdm(range(len(lats)), desc="Scoring pixels"):
        scores[i] = fast_scorer.score(lats[i], lons[i])

    # Apply colormap
    print("Applying colormap...")
    colors = cmap(scores)[:, :3]  # RGB without alpha

    # Fill output array
    for i, (x, y) in enumerate(zip(land_x, land_y)):
        for dy in range(sample_step):
            for dx in range(sample_step):
                if y + dy < height and x + dx < width:
                    output[y + dy, x + dx] = colors[i]

    # Save output
    output_img = Image.fromarray((output * 255).astype(np.uint8))
    output_img.save(output_path)
    print(f"Saved heatmap to {output_path}")

    return output_path


def main():
    """Main entry point."""
    print("Initializing scorer...")
    scorer = EuropeScorer(kernel_bandwidth_km=500.0)

    print("\nRendering heatmap (optimized)...")
    output = render_heatmap_fast(scorer, sample_step=1)

    print(f"\nDone! Output saved to: {output}")


if __name__ == "__main__":
    main()
