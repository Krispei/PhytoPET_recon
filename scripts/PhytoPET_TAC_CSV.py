import re
import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import csv

# ============================================================================
# CONFIGURATION PARAMETERS - All adjustable parameters in one place
# ============================================================================

# File paths
TIFF_FOLDER = "09182025/09182025_pproj_3min_40itr_mov"
OUTPUT_FILENAME = "09182025_TTP_None" #for saving SGV file
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
    "Top": (145, 177, 12),
    "Middle": (148, 181, 48),
    "Bottom": (150, 187, 77),
}

MIN_CLIP = 10
MAX_CLIP = 55
TIME_STEPS = 3

with open("TACS_RAW.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["ROI", "Timestamp", "Intensity", "Smoothened_Intensity"])


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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
                    labels.append(f"{position} ROOT")
                else:
                    sign = "+" if soil_label > 0 else ""
                    labels.append(f"{position} {DIRECTION_LABELS[direction]} {sign}{soil_label}mm")
    
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



# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main analysis pipeline with Monte Carlo error estimation."""
    roi_coords, roi_labels = generate_roi_coordinates()
    


    print(f"Extracting time courses for {len(roi_coords)} ROIs...")
    timecourses = [extract_voxel_timecourse(TIFF_FOLDER, coord)
                   for coord in roi_coords]


    print("writing values in csv...")
    for coord, label in zip(roi_coords, roi_labels):

        intensity_vals = extract_voxel_timecourse(TIFF_FOLDER, coord)
        smoothened_intensity_vals = gaussian_filter1d(intensity_vals, sigma=SIGMA)
        time = [i * TIME_STEPS for i in range(len(intensity_vals))]

        intensity_vals = intensity_vals[MIN_CLIP:MAX_CLIP]
        smoothened_intensity_vals = smoothened_intensity_vals[MIN_CLIP:MAX_CLIP]
        time = time[MIN_CLIP:MAX_CLIP]



        with open("TACS_RAW.csv", "a", newline="") as file:
            writer = csv.writer(file)

            for i in range(len(intensity_vals)):

                writer.writerow([label, time[i], intensity_vals[i], smoothened_intensity_vals[i]])


    print("Complete!")

if __name__ == "__main__":
    main()