import re
import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import math

# === Configuration ===
tiff_folder = "09182025/09182025_pproj_3min_40itr_mov"

root_positions = {
    "Top": (145, 177, 12),
    "Middle": (148, 181, 48),
    "Bottom": (150, 187, 77),
}

# === Plot Parameters ===
SAVE = True
CURRENT_ROOT = "Bottom"  # Options: "Top", "Middle", "Bottom"
CURRENT_DIR = "Y"  # Options: "X", "Y"
SMOOTHING = "Gaussian"  # Options: "None", "Gaussian"
DERIVATIVE = 1  # Options: 0 (raw), 1 (first derivative)
FRAME_MINUTES = 3
SIGMA = 0.9  # Gaussian smoothing parameter

# X-axis display settings
X_MIN = 10 # Minimum x value to display
X_MAX = 50  # Maximum x value to display
X_TICK_INTERVAL = 2  # Show tick every N frames

# Colors
top_colors = ["#00b9095d","#00b909b6","#00b909","#00b909a9","#00b90940"]
middle_colors = ["#007eb949", "#007eb9a3", "#007eb9", "#007eb9a3", "#007eb949"]
bottom_colors = ["#0016b941", "#0016b99b", "#0016b9", "#0016b99b", "#0016b941"]
colors = [top_colors, middle_colors, bottom_colors]

map_root_to_color = {
    "Top": 0,
    "Middle": 1,
    "Bottom": 2 
}

DEPTH_COLOR = colors[map_root_to_color[CURRENT_ROOT]] 
ROOT_COLOR = DEPTH_COLOR[2]
ROOT_LINEWIDTH = 2
ROOT_MARKER = 'o'
ROOT_MARKERSIZE = 6
ROOT_MARKEVERY = 1

SOIL_COLORS = [DEPTH_COLOR[1], DEPTH_COLOR[1], DEPTH_COLOR[0], DEPTH_COLOR[0]]
SOIL_LINEWIDTH = 2
SOIL_LINESTYLE = ['--', ":", "-.", '-']
SOIL_MARKER = [None]
SOIL_MARKERSIZE = 5
SOIL_MARKEVERY = 1

# === Utility Functions ===

def numerical_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def load_tiff_stack(tiff_path):
    return tifffile.imread(tiff_path)

def extract_voxel_curve(folder, coord):
    x, y, z = coord
    intensity_values = []
    #sorts the frames in their order "e.g. 0min, 3min, 6min, ..."
    tiff_files = sorted([f for f in os.listdir(folder) if f.lower().endswith((".tif", ".tiff"))],
                        key=numerical_sort_key)
    
    #gets the mean of the ROI intensity values 
    for fname in tiff_files:
        volume = load_tiff_stack(os.path.join(folder, fname))
        roi = volume[z:z+1, y:y+2, x:x+2]
        intensity_values.append(roi.mean())

    return intensity_values, tiff_files

def generate_soil_coords(root_coord, dists=[2, 4], dir="X"):
    X, Y, Z = root_coord
    coords, labels = [], []
    for dist in dists:
        if dir == "X":
            coords.append((X+dist, Y+dist, Z))
            labels.append(f"Soil +{round(np.sqrt(2*(dist/2)**2), 1)}mm")
            coords.append((X-dist, Y-dist, Z))
            labels.append(f"Soil -{round(np.sqrt(2*(dist/2)**2), 1)}mm")
        elif dir == "Y":
            coords.append((X-dist, Y+dist, Z))
            labels.append(f"Soil +{round(np.sqrt(2*(dist/2)**2), 1)}mm")
            coords.append((X+dist, Y-dist, Z))
            labels.append(f"Soil -{round(np.sqrt(2*(dist/2)**2), 1)}mm")
    return coords, labels

def apply_derivative(y, order=0, dt=1.0):
    y = np.asarray(y)
    if order == 0:
        return y
    elif order == 1:
        return np.gradient(y, dt)
    else:
        raise ValueError("order must be 0 or 1")

def apply_smoothing(data, smoothing_mode):
    if smoothing_mode == "None":
        return np.array(data)
    elif smoothing_mode == "Gaussian":
        return gaussian_filter1d(data, sigma=SIGMA)

# === Main Plot ===

def plot():
    root_coord = root_positions[CURRENT_ROOT]
    soil_coords, soil_labels = generate_soil_coords(root_coord, dir=CURRENT_DIR)

    root_curve, files = extract_voxel_curve(tiff_folder, root_coord)
    soil_curves = [extract_voxel_curve(tiff_folder, coord)[0] for coord in soil_coords]

    x = np.arange(len(files))
    fig, ax = plt.subplots(figsize=(12, 6))

    # Estimate per-frame noise

    # --- Root Curve ---
    y_root = apply_smoothing(root_curve, SMOOTHING)
    print(y_root)

    y_root = apply_derivative(y_root, order=DERIVATIVE, dt=FRAME_MINUTES)
    auc_root = np.trapz(y_root, x * FRAME_MINUTES)
    ax.plot(x, y_root, label=f'Root {CURRENT_ROOT} (AUC={auc_root:.1f})',
            lw=ROOT_LINEWIDTH, color=ROOT_COLOR,
            marker=ROOT_MARKER, markersize=ROOT_MARKERSIZE, markevery=ROOT_MARKEVERY)
    
    # --- Soil Curves ---
    for i, (y_soil, label) in enumerate(zip(soil_curves, soil_labels)):
        y = apply_smoothing(y_soil, SMOOTHING)
        y = apply_derivative(y, order=DERIVATIVE, dt=FRAME_MINUTES)
        auc = np.trapz(y, x * FRAME_MINUTES) if DERIVATIVE == 0 else None
        plot_label = f'{label} (AUC={auc:.1f})' if auc else label
        ax.plot(x, y, label=plot_label,
                linestyle=SOIL_LINESTYLE[i % len(SOIL_LINESTYLE)], linewidth=SOIL_LINEWIDTH,
                color=SOIL_COLORS[i % len(SOIL_COLORS)],
                marker=SOIL_MARKER[i % len(SOIL_MARKER)], markersize=SOIL_MARKERSIZE, markevery=SOIL_MARKEVERY)


    # --- Plot formatting ---
    deriv_text = {0: "None", 1: "1st"}[DERIVATIVE]
    ylabel = ["Intensity (a.u)", "Intensity change (per min)"][DERIVATIVE]

    ax.set_title(f"Time vs. Activity {CURRENT_ROOT} {CURRENT_DIR} - Smoothing: {SMOOTHING}, σ = {SIGMA}, Derivative: {deriv_text}", fontsize=14)
    ax.set_xlabel("Time Frame (min)", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xticks(np.arange(0, len(files), X_TICK_INTERVAL))
    ax.set_xticklabels([f"{i*FRAME_MINUTES}" for i in np.arange(0, len(files), X_TICK_INTERVAL)], rotation=45)
    ax.set_xlim(X_MIN, X_MAX)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.grid(axis='x', linestyle='--', alpha=0.6)
    plt.tick_params(axis='both', labelsize=14)  #
    ax.legend(fontsize=14)
    plt.tight_layout()
    if SAVE:
        plt.savefig(f"09182025_TAC_{CURRENT_ROOT}_{CURRENT_DIR}_DER{DERIVATIVE}_{SMOOTHING}", format="svg")

    plt.show()



if __name__ == "__main__":
    plot()
