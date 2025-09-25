import re
import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# === Configuration ===
tiff_folder = "SummerPHYTOPET/Plant_raw_data_St_10_40am/MOVIE_Plant_raw_data_St_10_40am_5min"
apply_smoothing = True
sigma = 0.6  # Standard deviation for Gaussian smoothing

root_voxels = []
soil_coords = []
soil_labels = []
labels = []

# List of root and soil voxel coordinates (X, Y, Z)
root_voxels = [
    (172,162,80),
    (178,157,43),
    (186,148,18)
]

labels = [
    "Top",
    "Middle",
    "Bottom"
]

def numerical_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def load_tiff_stack(tiff_path):
    return tifffile.imread(tiff_path)

def extract_voxel_curve(folder, coord):
    x, y, z = coord
    intensity_values = []

    tiff_files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(".tif") or f.lower().endswith(".tiff")
    ], key=numerical_sort_key)

    if not tiff_files:
        raise ValueError("No TIFF files found in the folder.")

    for fname in tiff_files:
        tiff_path = os.path.join(folder, fname)
        volume = load_tiff_stack(tiff_path)

        if x >= volume.shape[2] or y >= volume.shape[1] or z >= volume.shape[0]:
            raise IndexError(f"Voxel {coord} out of bounds for file {fname} with shape {volume.shape[::-1]}.")

        intensity = volume[z, y, x]
        intensity_values.append(intensity)

    return intensity_values, tiff_files

def gaussian_smoothing(data, sigma=1):
    return gaussian_filter1d(data, sigma=sigma)

def plot_multiple_curves(root_data, soil_data, filenames, smooth=True, sigma=1):
    plt.figure(figsize=(12, 6))
    x_plot = np.arange(len(filenames))

    for i, (coord, values) in enumerate(root_data):
        y = gaussian_smoothing(values, sigma=sigma) if smooth else np.array(values)
        auc = np.trapz(y)
        plt.plot(x_plot, y, label=f'Root {labels[i]} coord={coord} (AUC={auc:.1f})', alpha=1, linewidth=1.5, linestyle='-')
    
    '''
    soil_avg_sum = np.zeros(len(x_plot))
    for i, (coord, values) in enumerate(soil_data):
        y = gaussian_smoothing(values, sigma=sigma) if smooth else np.array(values)
        soil_avg_sum += y
        auc = np.trapz(y)
        plt.plot(x_plot, y, label=f'{soil_labels[i]} coord={coord} (AUC={auc:.1f})', linewidth=1.5, linestyle='--')
    '''
        

    title_suffix = f" (Gaussian σ={sigma})" if smooth else " (Raw)"
    plt.title(f"Root ROI Time Intensity Curves{title_suffix}")
    plt.xlabel("Time Frame")
    plt.ylabel("Intensity")
    plt.xticks(ticks=x_plot, labels=[f"{i*5} min" for i in x_plot], rotation=90)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    root_curves = []
    soil_curves = []

    for coord in root_voxels:
        values, files = extract_voxel_curve(tiff_folder, coord)
        auc = np.trapz(values)
        print(f"AUC for root voxel {coord}: {auc:.2f}")
        root_curves.append((coord, values))
    '''
    for coord in soil_coords:
        values, _ = extract_voxel_curve(tiff_folder, coord)
        auc = np.trapz(values)
        print(f"AUC for soil voxel {coord}: {auc:.2f}")
        soil_curves.append((coord, values))
    '''

    plot_multiple_curves(root_curves, soil_curves, files, smooth=apply_smoothing, sigma=sigma)
