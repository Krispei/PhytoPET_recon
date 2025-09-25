import re
import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt

# === Configuration ===
tiff_folder = "SummerPHYTOPET/Plant_raw_data_St_10_40am/MOVIE_Plant_raw_data_St_10_40am_5min/"  # Replace with your actual folder path
voxel_coord_1 = (155, 169, 66)  # (X, Y, Z) coordinate to extract time curve
voxel_coord_2 = (159, 166, 66)  # First voxel (X, Y, Z)

def numerical_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def load_tiff_stack(tiff_path):
    """Loads a 3D stack from a TIFF file."""
    return tifffile.imread(tiff_path)

def extract_voxel_curve(folder, coord):
    """Extracts the intensity values over time at a specific voxel coordinate."""
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
            raise IndexError(f"Voxel coordinate {coord} is out of bounds for file {fname} with shape {volume.shape[::-1]}.")

        intensity = volume[z, y, x]  # tifffile loads as (Z, Y, X)
        intensity_values.append(intensity)

    return intensity_values, tiff_files

def moving_average(data, window_size=3):
    """Computes the moving average of a 1D list or array."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_dual_curve(values1, coord1, auc1, values2, coord2, auc2, filenames, window_size=3):
    plt.figure(figsize=(10, 5))

    # Time axis
    x = np.arange(len(values1))
    x_avg = np.arange(window_size - 1, len(values1))

    # Compute moving averages
    values1_avg = moving_average(values1, window_size)
    values2_avg = moving_average(values2, window_size)

    # Plot raw curves
    plt.plot(x, values1, marker='o', linestyle='--', alpha=0.4, label=f'Root {coord1} (raw)')
    plt.plot(x, values2, marker='o', linestyle='--', alpha=0.4, label=f'Soil {coord2} (raw)')

    # Plot smoothed curves
    plt.plot(x_avg, values1_avg, color='blue', linewidth=2, label=f'Root moving average {coord1} (avg)')
    plt.plot(x_avg, values2_avg, color='red', linewidth=2, label=f'Soil moving average {coord2} (avg)')

    plt.title("07/16/2025 Data (with Moving Average)")
    plt.xlabel("Time Frame")
    plt.ylabel("Intensity")
    plt.grid(True)
    plt.legend()

    # Adjust x-axis labels to match time
    plt.xticks(ticks=x, labels=[f"{i*5} min" for i in range(len(filenames))], rotation=90)

    plt.tight_layout()
    plt.show()

# Call plot function with a desired window size
if __name__ == "__main__":
    values1, files = extract_voxel_curve(tiff_folder, voxel_coord_1)
    values2, _ = extract_voxel_curve(tiff_folder, voxel_coord_2)

    auc1 = np.trapz(values1)
    auc2 = np.trapz(values2)

    print(f"AUC for voxel {voxel_coord_1}: {auc1:.2f}")
    print(f"AUC for voxel {voxel_coord_2}: {auc2:.2f}")

    plot_dual_curve(values1, voxel_coord_1, auc1, values2, voxel_coord_2, auc2, files, window_size=3)
