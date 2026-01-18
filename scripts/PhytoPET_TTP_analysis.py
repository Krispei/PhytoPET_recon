import re
import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Patch
import csv
import math

# ============================================================================
# CONFIGURATION PARAMETERS - All adjustable parameters in one place
# ============================================================================

# File paths
TIFF_FOLDER = "07162025/0716025_pproj_3min_65mm_040itr"
OUTPUT_FILENAME = "09182025_TTP_Roots_Gaussian" #for saving SGV file
SAVE = False # Set to true to save SGV

# Smoothing settings
APPLY_SMOOTHING = True
SIGMA = 0.9  # Standard deviation for Gaussian smoothing

# Analysis directions (0=X, 1=Y, 2=Z)
DIRECTIONS = (0, 1)
DIRECTION_LABELS = ("X", "Y")

# Soil extensions and labels (in mm, multiplied by 2 for pixels)
SOIL_EXTENSIONS = (-4, -2, 0, 2, 4)  # in mm
SOIL_LABELS = (-2.8, -1.4, 0, 1.4, 2.8)  # in mm

# ROI positions (X, Y, Z coordinates)
ROI_POSITIONS = {
    "Top": (54,80,15),
    "Middle": (59,96,34),
    "Bottom": (57,116,48)
}

# Plot settings
FIGURE_SIZE = (14, 7)
PLOT_ONLY_ROOTS = False
Y_LIMIT = 40
GROUP_SIZE = 5
GAP_WIDTH = 1
Y_TICK_INTERVAL = 2  # Show every Nth tick
TIME_MULTIPLIER = 3  # Convert indices to minutes
text_rotation = 45
text_align = "right"

# Color schemes for each depth
COLORS = {
    "Top": ["#00b90940", "#00b909a9", "#00b909", "#00b909a9", "#00b90940"],
    "Middle": ["#007eb949", "#007eb9a3", "#007eb9", "#007eb9a3", "#007eb949"],
    "Bottom": ["#0016b941", "#0016b99b", "#0016b9", "#0016b99b", "#0016b941"],
}

# Legend colors
LEGEND_COLORS = {
    "Top": "#00b909",
    "Middle": "#007eb9",
    "Bottom": "#0016b9",
}

# ERROR CORRECTION PARAMETERS
MIN_CLIP = 10
MAX_CLIP = 60

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ttp_uncertainty_bootstrap(time, raw_data, smoothed_data, sigma_smooth, 
                              n_iter=1000, random_state=None):
    """
    Estimate time-to-peak (TTP) uncertainty using empirical residual bootstrap.
    
    Parameters
    ----------
    time : array-like
        Time points (e.g., mid-frame times in minutes or seconds).
    raw_data : array-like
        Original voxel/ROI intensity values before smoothing.
    smoothed_data : array-like
        Gaussian-smoothed intensity values.
    sigma_smooth : float
        Standard deviation (in frames) of the Gaussian smoothing kernel applied.
    n_iter : int
        Number of bootstrap iterations.
    random_state : int or None
        Seed for reproducibility.
    
    Returns
    -------
    results : dict
        Dictionary with keys:
        - 'ttp_peak': measured peak from smoothed_data
        - 'ttp_std': bootstrap standard deviation
        - 'ttp_ci95': 95% confidence interval
        - 'ttp_samples': all sampled TTPs
    """
    rng = np.random.default_rng(random_state)
    
    # Clip relevant window if needed
    raw_data_clip = raw_data[MIN_CLIP:MAX_CLIP]
    smoothed_clip = smoothed_data[MIN_CLIP:MAX_CLIP]
    time_clip = time[MIN_CLIP:MAX_CLIP]
    
    # Residuals (empirical noise)
    residuals = raw_data_clip - smoothed_clip
    residuals_nonzero = residuals[residuals != 0]
    if len(residuals_nonzero) == 0:
        residuals_nonzero = np.zeros_like(residuals)
    
    # Measured TTP from smoothed curve
    peak_idx = np.argmax(smoothed_clip)
    ttp_peak = time_clip[peak_idx]
    
    # Bootstrap resampling
    ttp_samples = []
    for _ in range(n_iter):
        # Sample residuals with replacement (empirical)
        sampled_noise = rng.choice(residuals_nonzero, size=len(smoothed_clip), replace=True)
        noisy_curve = smoothed_clip + sampled_noise
        # Re-smooth to mimic processing
        re_smoothed = gaussian_filter1d(noisy_curve, sigma=sigma_smooth)
        peak_idx_iter = np.argmax(re_smoothed)
        ttp_samples.append(time_clip[peak_idx_iter])
    
    ttp_samples = np.array(ttp_samples)
    
    return {
        "ttp_peak": ttp_peak,
        "ttp_std": np.std(ttp_samples),
        "ttp_ci95": np.percentile(ttp_samples, [2.5, 97.5]),
        "ttp_samples": ttp_samples
    }


def numerical_sort_key(s):
    """Sort filenames with numerical components correctly."""
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', s)]


def generate_roi_coordinates():
    """Generate all ROI coordinates and labels based on configuration."""
    coords = []
    labels = []
    
    for position in ROI_POSITIONS:
        for direction in DIRECTIONS:
            for i, extend in enumerate(SOIL_EXTENSIONS):
                # Calculate new coordinates
                x, y, z = ROI_POSITIONS[position]
                if direction == 0:  # X direction
                    coords.append((x + extend, y + extend, z))
                else:  # Y direction
                    coords.append((x - extend, y + extend, z))
                
                # Generate label
                soil_label = SOIL_LABELS[i]
                if soil_label == 0:
                    labels.append(f"{position} root")
                else:
                    sign = "+" if soil_label > 0 else ""
                    labels.append(f"{DIRECTION_LABELS[direction]} {sign}{soil_label}mm")
    
    return coords, labels


def load_tiff_files(folder):
    """Load and sort TIFF files from folder. Cached for performance."""
    if not hasattr(load_tiff_files, '_cache'):
        tiff_files = [f for f in os.listdir(folder) 
                      if f.lower().endswith((".tif", ".tiff"))]
        load_tiff_files._cache = sorted(tiff_files, key=numerical_sort_key)
    return load_tiff_files._cache


def extract_voxel_timecourse(folder, coord):
    """Extract intensity time course for a single voxel coordinate."""
    x, y, z = coord
    tiff_files = load_tiff_files(folder)
    
    # Pre-allocate array for better performance
    intensity_values = np.empty(len(tiff_files), dtype=np.float32)
    
    for i, fname in enumerate(tiff_files):
        tiff_path = os.path.join(folder, fname)
        volume = tifffile.imread(tiff_path)
        
        # Bounds check
        if x + 1 >= volume.shape[2] or y + 1 >= volume.shape[1] or z + 1 >= volume.shape[0]:
            raise IndexError(
                f"ROI {coord} out of bounds for {fname} with shape {volume.shape[::-1]}"
            )
        
        # Extract 2x2x1 ROI and compute mean (vectorized operation)
        intensity_values[i] = volume[z:z+1, y:y+2, x:x+2].mean()
    
    return intensity_values


def generate_plot_colors(roi_positions):
    """Generate color list matching the ROI order."""
    return [color 
            for position in roi_positions 
            for color in COLORS[position] * len(DIRECTIONS)]


def plot_time_to_peak(ttp_frames, ttp_stds_frames, labels):
    """
    Create a bar plot of TTP in minutes with 1σ error bars.
    Converts frame indices to minutes using TIME_MULTIPLIER.
    """
    # Convert frame indices and uncertainties to minutes
    ttp_minutes = np.array(ttp_frames) * TIME_MULTIPLIER
    ttp_stds_minutes = np.array(ttp_stds_frames) * TIME_MULTIPLIER

    indices = np.arange(len(ttp_minutes))
    x_positions = indices + GAP_WIDTH * (indices // GROUP_SIZE)
    
    colors = generate_plot_colors(ROI_POSITIONS.keys())

    #print(ttp_minutes)

    if PLOT_ONLY_ROOTS:
        
        ttp_minutes = [ttp_minutes[2], ttp_minutes[12], ttp_minutes[22]]
        ttp_stds_minutes = [ttp_stds_minutes[2],ttp_stds_minutes[12],ttp_stds_minutes[22]]
        indices = [indices[2], indices[12], indices[22]]
        x_positions = [x_positions[2], x_positions[12], x_positions[22]]
        colors = [colors[2], colors[12], colors[22]]
        labels = ["Top", "Middle", "Bottom"]        
        x_positions = [0,1.5,3]

        text_rotation = 0
        text_align = "center"
    else:
        text_rotation = 45
        text_align = "right"


    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.bar(
        x_positions, ttp_minutes, color=colors, width=1,
        yerr=ttp_stds_minutes, capsize=3, ecolor='black', error_kw=dict(lw=1, capthick=1)
    )
    
    print(type(ttp_minutes), type(ttp_stds_minutes))

    # Configure y-axis (in minutes)
    max_y = int(np.ceil(max(ttp_minutes + ttp_stds_minutes)))  # round up to integer
    yticks = np.arange(0, max_y + 1, TIME_MULTIPLIER)          # integer ticks

    # Show every other tick label
    ax.set_yticks(yticks)


    # Styling
    # Only draw grid lines for the ticks that are labeled (visible)
    visible_grid = [label for i, label in enumerate(yticks) if i % 1 == 0]
    ax.yaxis.grid(True, which='major', linestyle='-', alpha=0.6)
    ax.set_yticks(visible_grid, minor=False)  # ensures grid aligns with visible labels
    ax.set_ylim(Y_LIMIT, 72)

    plt.xticks(x_positions, labels, rotation=text_rotation, ha=text_align)
    ax.set_xlabel("Regions of Interest (ROIs)", fontsize=14)
    ax.set_ylabel("Time to Peak (min)", fontsize=14)
    ax.set_title("Time to Peak Across Root Sample (±1σ)", fontsize=14)
    ax.tick_params(axis='both', labelsize=14)

    legend_patches = [
        Patch(facecolor=LEGEND_COLORS[pos], label=f"{pos} (Z={ROI_POSITIONS[pos][2]})")
        for pos in ROI_POSITIONS
    ]
    ax.legend(handles=legend_patches, title="Root Depth", fontsize=18)

    plt.tight_layout()
    if SAVE:
        plt.savefig(f"{OUTPUT_FILENAME}.svg", format="svg")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main analysis pipeline with Monte Carlo error estimation."""
    roi_coords, roi_labels = generate_roi_coordinates()

    print(f"Extracting time courses for {len(roi_coords)} ROIs...")
    timecourses = [extract_voxel_timecourse(TIFF_FOLDER, coord)
                   for coord in roi_coords]

     # Compute smoothed signals and TTP uncertainty
    print("Computing Monte Carlo time-to-peak uncertainties...")
    time = np.arange(len(timecourses[0]))  # time indices in frames
    ttp_means, ttp_stds, ttp_ci95 = [], [], []

    for raw in timecourses:
        smoothed = gaussian_filter1d(raw, sigma=SIGMA)

        # --- Actual measured peak from smoothed data ---
        peak_idx = np.argmax(smoothed)
        ttp_peak = time[peak_idx]  # actual TTP in frames

        # --- Monte Carlo uncertainty ---
        result = ttp_uncertainty_bootstrap(time, raw, smoothed, sigma_smooth=SIGMA)

        # --- Use measured TTP with Monte Carlo uncertainty ---
        ttp_means.append(ttp_peak)
        ttp_stds.append(result["ttp_std"])  # in frame units
        ttp_ci95.append(result["ttp_ci95"])



    print("Generating plot with error bars...")
    plot_time_to_peak(ttp_means, ttp_stds, roi_labels)

    if SAVE:
        print(f"Plot saved as {OUTPUT_FILENAME}.svg")
    else:
        print("Plot displayed (SAVE=False)")
        
   

    with open("TTP_STATS.csv", "w", newline="") as file:
        writer = csv.writer(file)
        
        writer.writerow(["ROI", "TTP", "1STD", "95CI"])

        for i in range(len(roi_labels)):
            writer.writerow([roi_labels[i], ttp_means[i] * TIME_MULTIPLIER, (list(ttp_stds)[i] * TIME_MULTIPLIER), list(ttp_ci95)[i] * TIME_MULTIPLIER])




if __name__ == "__main__":
    main()