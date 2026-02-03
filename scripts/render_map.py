"""Unified renderer for Europe classification maps.

Generates:
1. SVM Probability Heatmap (europe_focus_svm.png)
2. Linear Probability Heatmap (europe_focus_linear.png)
3. Binary Classification Maps (europe_focus_*_binary.png)
4. Contour Probability Map (europe_svm_contours.png)

Usage:
    uv run python scripts/render_map.py [--full]
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from europe_or_not.scorer import MultiClassifierScorer

# Region bounds (should match what scorer expects/training data covers roughly)
LON_MIN, LON_MAX = -30, 60
LAT_MIN, LAT_MAX = 30, 75

OUTPUT_DIR = Path(__file__).parent.parent
RASTER_PATH = OUTPUT_DIR / "equi_hires.png"


def crop_to_region(img, full_width, full_height):
    """Crop image to the Europe focus region."""
    x_min = int((LON_MIN + 180) / 360 * full_width)
    x_max = int((LON_MAX + 180) / 360 * full_width)
    y_min = int((90 - LAT_MAX) / 180 * full_height)
    y_max = int((90 - LAT_MIN) / 180 * full_height)
    return img.crop((x_min, y_min, x_max, y_max))


def render_maps(full_res: bool = False):
    """Render all map artifacts."""
    # Step size: 1 for full res, 4 for preview
    sample_step = 1 if full_res else 4
    suffix = "" if full_res else "_lowres"
    
    print(f"Initializing Scorer (Full Res: {full_res})...")
    scorer = MultiClassifierScorer()
    
    # Load and crop raster
    if not RASTER_PATH.exists():
        print(f"Error: {RASTER_PATH} not found.")
        return

    base_img = Image.open(RASTER_PATH).convert("RGBA")
    cropped = crop_to_region(base_img, *base_img.size)
    width, height = cropped.size
    
    gray = np.array(cropped.convert('L'))
    is_land = gray < 128
    
    if sample_step > 1:
        land_y, land_x = np.where(is_land[::sample_step, ::sample_step])
        land_y, land_x = land_y * sample_step, land_x * sample_step
    else:
        land_y, land_x = np.where(is_land)
    
    print(f"Scoring {len(land_x):,} land pixels (step={sample_step})...")
    
    lons = LON_MIN + (land_x / width) * (LON_MAX - LON_MIN)
    lats = LAT_MAX - (land_y / height) * (LAT_MAX - LAT_MIN)
    
    # Score pixels
    svm_scores = np.zeros(len(lats))
    linear_scores = np.zeros(len(lats))
    
    for i in tqdm(range(len(lats)), desc="Scoring"):
        svm_scores[i], linear_scores[i] = scorer.score_both(lats[i], lons[i])
    
    # Grid for contours/images
    # Create normalized outputs
    cmap = plt.cm.plasma
    
    # 1. Standard Heatmaps and Binary Maps
    yellow = np.array([0.94, 0.97, 0.13])
    purple = np.array([0.05, 0.03, 0.53])
    
    for name, scores in [("svm", svm_scores), ("linear", linear_scores)]:
        # Probability Map
        output = np.ones((height, width, 3), dtype=np.float32)
        # Fill colors
        colors = cmap(scores)[:, :3]
        
        # Binary Map
        output_bin = np.ones((height, width, 3), dtype=np.float32)
        colors_bin = np.where(scores[:, None] >= 0.5, yellow, purple)
        
        # Fill pixels (splatting for low-res)
        for i, (x, y) in enumerate(zip(land_x, land_y)):
            for dy in range(sample_step):
                for dx in range(sample_step):
                    if y + dy < height and x + dx < width:
                        output[y + dy, x + dx] = colors[i]
                        output_bin[y + dy, x + dx] = colors_bin[i]

        path_prob = OUTPUT_DIR / f"europe_focus_{name}{suffix}.png"
        path_bin = OUTPUT_DIR / f"europe_focus_{name}_binary{suffix}.png"
        
        Image.fromarray((output * 255).astype(np.uint8)).save(path_prob)
        Image.fromarray((output_bin * 255).astype(np.uint8)).save(path_bin)
        print(f"Saved {path_prob}")
        print(f"Saved {path_bin}")

    # 2. Contour Map (SVM Only) - Only makes sense to run matplotlib contour gen
    # We reconstruct the grid
    print("Generating contours...")
    grid_h = height // sample_step + 1
    grid_w = width // sample_step + 1
    score_grid = np.full((grid_h, grid_w), np.nan)
    
    for i, (x, y) in enumerate(zip(land_x, land_y)):
        grid_y, grid_x = y // sample_step, x // sample_step
        if grid_y < grid_h and grid_x < grid_w:
            score_grid[grid_y, grid_x] = svm_scores[i]
            
    fig, ax = plt.subplots(figsize=(14, 7))
    extent = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]
    im = ax.imshow(score_grid, cmap=cmap, extent=extent, origin='upper', vmin=0, vmax=1, aspect='auto')
    
    lon_grid = np.linspace(LON_MIN, LON_MAX, grid_w)
    lat_grid = np.linspace(LAT_MAX, LAT_MIN, grid_h)
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    
    levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    cs = ax.contour(LON, LAT, score_grid, levels=levels, colors='white', linewidths=1, linestyles='solid')
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('SVM Europe Probability')
    plt.colorbar(im, ax=ax, label='P(Europe)')
    
    path_contour = OUTPUT_DIR / f"europe_svm_contours{suffix}.png"
    plt.tight_layout()
    plt.savefig(path_contour, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path_contour}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render Europe classification maps")
    parser.add_argument("--full", action="store_true", help="Render full resolution (slower)")
    args = parser.parse_args()
    
    render_maps(full_res=args.full)
