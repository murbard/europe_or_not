"""Render equirectangular heatmap of Europe scores."""

import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm

from europe_or_not.scorer import EuropeScorer


BASE_MAP = Path(__file__).parent.parent / "equi.png"
OUTPUT_PATH = Path(__file__).parent.parent / "europe_heatmap.png"


def is_land(pixel: tuple) -> bool:
    """Check if a pixel is land based on alpha channel."""
    # Land has alpha > 0, ocean has alpha = 0 (transparent)
    if len(pixel) >= 4:
        return pixel[3] > 0  # alpha channel
    # Fallback for RGB
    r, g, b = pixel[:3]
    return not (r > 240 and g > 240 and b > 240)


def pixel_to_coords(x: int, y: int, width: int, height: int) -> tuple[float, float]:
    """Convert pixel coordinates to lat/lon.
    
    Equirectangular projection:
    - x maps linearly to longitude (-180 to 180)
    - y maps linearly to latitude (90 to -90, top to bottom)
    """
    lon = (x / width) * 360 - 180
    lat = 90 - (y / height) * 180
    return lat, lon


def create_heatmap_colormap():
    """Create a colormap from blue (not Europe) to red (Europe)."""
    # Blue -> Cyan -> Green -> Yellow -> Orange -> Red
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


def render_heatmap(
    scorer: EuropeScorer,
    base_map_path: Path = BASE_MAP,
    output_path: Path = OUTPUT_PATH,
    sample_step: int = 1,
) -> None:
    """Render Europe score heatmap over land areas.
    
    Args:
        scorer: EuropeScorer instance.
        base_map_path: Path to equirectangular base map.
        output_path: Where to save the output.
        sample_step: Sample every Nth pixel (for faster rendering).
    """
    # Load base map
    base_img = Image.open(base_map_path).convert("RGBA")
    width, height = base_img.size
    print(f"Base map size: {width}x{height}")
    
    # Create output array
    output = np.ones((height, width, 3), dtype=np.float32)  # Start with white (ocean)
    
    # Get colormap
    cmap = create_heatmap_colormap()
    
    # Process each pixel
    land_pixels = []
    
    print("Finding land pixels...")
    for y in range(0, height, sample_step):
        for x in range(0, width, sample_step):
            pixel = base_img.getpixel((x, y))
            if is_land(pixel):
                land_pixels.append((x, y))
    
    print(f"Found {len(land_pixels)} land pixels to process")
    
    # Compute scores for land pixels
    print("Computing Europe scores...")
    for x, y in tqdm(land_pixels, desc="Scoring pixels"):
        lat, lon = pixel_to_coords(x, y, width, height)
        score = scorer.score(lat, lon)
        
        # Get color from colormap
        color = cmap(score)[:3]  # RGB without alpha
        
        # Fill in the sampled region
        for dy in range(sample_step):
            for dx in range(sample_step):
                if y + dy < height and x + dx < width:
                    output[y + dy, x + dx] = color
    
    # Save output
    output_img = Image.fromarray((output * 255).astype(np.uint8))
    output_img.save(output_path)
    print(f"Saved heatmap to {output_path}")
    
    return output_path


def main():
    """Main entry point."""
    print("Initializing scorer...")
    scorer = EuropeScorer(kernel_bandwidth_km=500.0)
    
    print("\nRendering heatmap...")
    # Use sample_step=2 for faster rendering (can set to 1 for full resolution)
    output = render_heatmap(scorer, sample_step=2)
    
    print(f"\nDone! Output saved to: {output}")


if __name__ == "__main__":
    main()
