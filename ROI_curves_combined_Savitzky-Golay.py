import re
import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# === Configuration ===
tiff_folder = "SummerPHYTOPET/Plant_raw_data_St_10_40am/MOVIE_Plant_raw_data_St_10_40am_5min"
apply_smoothing = True
window_length = 5   # Must be odd and <= number of time points
polyorder = 2       # Polynomial order for Savitzky-Golay

root_voxels = []
soil_coords = []
soil_labels = []
labels = []

# List of root and soil voxel coordinates (X, Y, Z)
root_pos = [
    (172,163,83),  # Top
    (178,157,43),  # Middle
    (182,152,23)   # Bottom
]

def root(position="top"):
    if position == "top":
        root_voxels.append(root_pos[0])
        labels.append(position)
    elif position == "middle":
        root_voxels.append(root_pos[1])
        labels.append(position)
    elif position == "bottom":
        root_voxels.append(root_pos[2])
        labels.append(position)

def soil(root_coord=(50,50,50), dists=[2,4,8], dir="X"):
    X, Y, Z = root_coord
    for dist in dists: 
        if dir == "X":
            soil_coords.extend([(X+dist, Y, Z), (X-dist, Y, Z)])
            soil_labels.extend([f"Soil +{dist/2}mm", f"Soil -{dist/2}mm"])
        elif dir == "Y":
            soil_coords.extend([(X, Y+dist, Z), (X, Y-dist, Z)])
            soil_labels.extend([f"Soil +{dist/2}mm", f"Soil -{dist/2}mm"])
        elif dir == "Z":
            soil_coords.extend([(X, Y, Z+dist), (X, Y, Z-dist)])
            soil_labels.extend([f"Soil +{dist/2}mm", f"Soil -{dist/2}mm"])

root(position="top")
soil(root_coord=root_voxels[0], dir="X")

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

def savgol_smoothing(data, window_length=5, polyorder=2):
    if len(data) < window_length:
        raise ValueError(f"Not enough time points ({len(data)}) for window_length={window_length}")
    return savgol_filter(data, window_length=window_length, polyorder=polyorder)

def plot_multiple_curves(root_data, soil_data, filenames, smooth=True, window_length=5, polyorder=2):
    plt.figure(figsize=(12, 6))
    x_plot = np.arange(len(filenames))

    root_avg_sum = np.zeros(len(x_plot))
    for i, (coord, values) in enumerate(root_data):
        y = savgol_smoothing(values, window_length, polyorder) if smooth else np.array(values)
        root_avg_sum += y
        auc = np.trapz(y)
        plt.plot(x_plot, y, label=f'Root {labels[i]} coord={coord} (AUC={auc:.1f})', alpha=0.4, linewidth=1.5, linestyle='-')

    soil_avg_sum = np.zeros(len(x_plot))
    for i, (coord, values) in enumerate(soil_data):
        y = savgol_smoothing(values, window_length, polyorder) if smooth else np.array(values)
        soil_avg_sum += y
        auc = np.trapz(y)
        plt.plot(x_plot, y, label=f'{soil_labels[i]} coord={coord} (AUC={auc:.1f})', linewidth=1.5, linestyle='--')

    title_suffix = f" (Savitzky-Golay, window={window_length}, order={polyorder})" if smooth else " (Raw)"
    plt.title(f"Voxel Time Intensity Curves{title_suffix}")
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

    for coord in soil_coords:
        values, _ = extract_voxel_curve(tiff_folder, coord)
        auc = np.trapz(values)
        print(f"AUC for soil voxel {coord}: {auc:.2f}")
        soil_curves.append((coord, values))

    plot_multiple_curves(
        root_curves, 
        soil_curves, 
        files, 
        smooth=apply_smoothing, 
        window_length=window_length, 
        polyorder=polyorder
    )
