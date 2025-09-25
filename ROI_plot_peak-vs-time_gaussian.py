import re
import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# === Configuration ===
tiff_folder = "SummerPHYTOPET/plant_raw_noglucose_07252025/MOVIE_07252025_Dyn_5min_2"
apply_smoothing = True
sigma = 1  # Standard deviation for Gaussian smoothing

directions = (0, 1) #0 = x, 1 = y, 2 = z
direction_list = ("X", "Y")
soil_ext = (-4, -2, 0, 2, 4) #in mm, times two to get pixels
soil_labels = (-2.8, -1.4, 0, 1.4, 2.8) #in mm, times two to get pixels


roi_coords = []
labels = []

# List of root and soil voxel coordinates (X, Y, Z)
roi_pos = {
    "Top": (163,151,81),
    "Middle": (161,131,61),
    "Bottom": (168,111,37)
}

'''
    "top": (171,161,80),
    "middle": (176,158,56),
    "bottom": (186,148,18)
'''


for position in roi_pos:
    for direction in directions:
        for i in range(len(soil_ext)):
            extend = soil_ext[i]

            new_soil = list(roi_pos[position])
            if direction == 0:
                new_soil[0] += extend * 1 #for voxel size 0.5
                new_soil[1] += extend * 1 #for voxel size 0.5
            else:
                new_soil[0] -= extend * 1 #for voxel size 0.5
                new_soil[1] += extend * 1 #for voxel size 0.5

            roi_coords.append(new_soil)
            labels.append(f"{position} {direction_list[direction]} {soil_labels[i]}mm")
            
            #print(f'{new_soil} {labels[-1]}')


num_sections = 3
bars_per_section = 10

def peak_iterate(y):

    max_val = 0
    max_index = 0

    for index,value in enumerate(y):

        if index < 10 or index > 17: 
            continue

        if (value > max_val):
            max_val = value
            max_index = index
    
    return max_index if max_val >= 3  else 0

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

    for fname in tiff_files:
        tiff_path = os.path.join(folder, fname)
        volume = load_tiff_stack(tiff_path)

        # Bounds check
        if x+1 >= volume.shape[2] or y+1 >= volume.shape[1] or z+1 >= volume.shape[0]:
            raise IndexError(f"ROI {coord} to {(x+1, y+1, z+1)} out of bounds for file {fname} with shape {volume.shape[::-1]}.")

        # Extract 2x2x2 region and compute mean
        roi = volume[z:z+1, y:y+2, x:x+2]  # shape 
        mean_intensity = roi.mean()
        intensity_values.append(mean_intensity)

    return intensity_values, tiff_files

def gaussian_smoothing(data, sigma=0.7):
    return gaussian_filter1d(data, sigma=sigma)

def find_peaks(roi_data, smooth=True, sigma=1):

    peak_list = []

    for coord, values in roi_data:
        y = gaussian_smoothing(values, sigma=sigma) if smooth else np.array(values)
        peak_list.append(peak_iterate(y))
    
    #peak_list = peak_process(peak_list)

    return(peak_list)

def plot(data):
    group_size = 5  # 7 bars per group
    gap_width = 1   # Width of gap between groups

    # Define 7 distinct colors
    color_cycle = ['#1f77b4', '#ff7f0e', 
                   '#d62728', '#9467bd', '#e377c2']

    # Compute x positions with gaps
    x = []
    colors = []

    for i in range(len(data)):
        group_index = i // group_size
        x_pos = i + gap_width * group_index
        x.append(x_pos)
        colors.append(color_cycle[i % group_size])  # Cycle through the colors

    # Plot
    plt.figure(figsize=(14, 6))
    plt.bar(x, data, color=colors)

    # Set y-axis ticks incrementing by 1
    max_y = max(data)
    yticks = range(0, max_y + 2)  # Tick positions
    ylabels = [f"{tick * 5}" for tick in yticks]  # Labels scaled and suffixed
    plt.yticks(yticks, ylabels)
    
    # Gridlines
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # Adjust x-ticks to match bar positions
    plt.xticks(x, labels, rotation=90)

    plt.ylabel('Peak Tim (min)')
    plt.title('Peak Timing Across ROIs')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    root_curves = []
    soil_curves = []

    for coord in roi_coords:
        values, files = extract_voxel_curve(tiff_folder, coord)
        root_curves.append((coord, values))

    peaks = find_peaks(roi_data=root_curves, smooth=apply_smoothing, sigma=sigma)
    plot(peaks)
