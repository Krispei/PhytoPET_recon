import re
import os
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter, find_peaks

# === Configuration ===
tiff_folder = "SummerPHYTOPET/plant_raw_noglucose_07252025/MOVIE_07252025_Dyn_5min_2"

root_positions = {
    "Top": (163,151,81),
    "Middle": (161,131,61),
    "Bottom": (168,111,37)

    #"Top": (164, 154, 84),
    #"Middle": (162,130,60),
    #"Bottom": (168,112,38)
}

smoothing_modes = ["None", "Moving Average", "Gaussian", "Savitzky-Golay"]

current_root = "Top"
current_dir = "X"
current_smoothing = "None"
current_derivative = 0
frame_minutes = 5
window_size = 3
polyorder = 2
sigma = 1

soil_coords = []
soil_labels = []
soil_dirs = []

def numerical_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def load_tiff_stack(tiff_path):
    return tifffile.imread(tiff_path)

def extract_voxel_curve(folder, coord, ROI_type, soildir):
    x, y, z = coord
    intensity_values = []

    tiff_files = sorted([
        f for f in os.listdir(folder)
        if f.lower().endswith(".tif") or f.lower().endswith(".tiff")
    ], key=numerical_sort_key)

    for fname in tiff_files:
        tiff_path = os.path.join(folder, fname)
        volume = load_tiff_stack(tiff_path)

        if ROI_type == "Root":
            roi = volume[z:z+1, y:y+2, x:x+2]
        elif ROI_type == "Soil":
            if soildir == "X":
                roi = volume[z:z+1, y:y+2, x:x+2]
            elif soildir == "Y":
                roi = volume[z:z+1, y:y+2, x:x+2]
            else:
                raise ValueError("Invalid soil direction")
        else:
            raise ValueError("Invalid ROI_type")

        mean_intensity = roi.mean()
        intensity_values.append(mean_intensity)

    return intensity_values, tiff_files

def generate_soil_coords(root_coord, dists=[4,2], dir="X"):
    print(dists)
    X, Y, Z = root_coord
    coords, label_dir, labels = [], [], []

    for dist in dists:
        if dir == "X":
            coords.append((X+dist, Y+dist, Z)); label_dir.append("X"); labels.append(f"Soil +{dist/2}mm")
            coords.append((X-dist, Y-dist, Z)); label_dir.append("X"); labels.append(f"Soil -{dist/2}mm")
        elif dir == "Y":
            coords.append((X-dist, Y+dist, Z)); label_dir.append("Y"); labels.append(f"Soil +{dist/2}mm")
            coords.append((X+dist, Y-dist, Z)); label_dir.append("Y"); labels.append(f"Soil -{dist/2}mm")
    return coords, label_dir, labels

def apply_derivative(y, order=0, dt=1.0):
    y = np.asarray(y)
    if order == 0:
        return y
    elif order == 1:
        return np.gradient(y, dt)
    elif order == 2:
        return np.gradient(np.gradient(y, dt), dt)
    else:
        raise ValueError("order must be 0, 1, or 2")

def apply_smoothing(data):
    if current_smoothing == "None":
        return np.array(data)
    elif current_smoothing == "Moving Average":
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    elif current_smoothing == "Gaussian":
        return gaussian_filter1d(data, sigma=sigma)
    elif current_smoothing == "Savitzky-Golay":
        return savgol_filter(data, window_length=window_size, polyorder=polyorder)

def find_peak_value(data, x_values, value_min=50, value_max=100, threshold=3):

    max_value = 0
    max_time = 0

    for i in range(value_min//5, value_max//5):
        if data[i] >= max_value:
            max_value = data[i]
            max_time = x_values[i]
    
    return (max_value, max_time) if max_value > threshold else (0,0)

def plot_peaks():
    """Create a bar chart showing peak times for all root sections and distances"""
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(bottom=0.2, top=0.9)
    
    root_names = ["Top", "Middle", "Bottom"]
    distances = [-2, -1, 0, 1, 2]  # in mm
    distance_labels = ["-2.8mm", "-1.4mm", "Root", "+1.4mm", "+2.8mm"]  # Fixed labels
    
    # Colors for each distance
    distance_colors = ['#ff9999', '#ffcccc', '#4472C4', '#ccddff', '#99bbff']
    
    # Collect all peak times
    all_peak_times = []
    section_labels = []
    colors_list = []
    
    for root_idx, root_name in enumerate(root_names):
        root_coord = root_positions[root_name]
        
        # Generate soil coordinates for this root
        soil_coords_local, soil_dirs_local, soil_labels_local = generate_soil_coords(
            root_coord, dists=[4, 2], dir=current_dir
        )
        
        # Extract curves for root and soil
        root_curve, files = extract_voxel_curve(tiff_folder, root_coord, "Root", "X")
        
        # Create time axis
        x = np.arange(len(files))
        
        # Process root curve
        y_root = apply_smoothing(root_curve)
        y_root = apply_derivative(y_root, order=current_derivative, dt=frame_minutes)
        x_plot = x[window_size-1:] if current_smoothing == "Moving Average" else x
        x_plot_time = x_plot * frame_minutes
        
        # Process soil curves - organize by distance
        soil_curves = []
        for i in range(len(soil_coords_local)):
            soil_curve, _ = extract_voxel_curve(tiff_folder, soil_coords_local[i], "Soil", soil_dirs_local[i])
            y_soil = apply_smoothing(soil_curve)
            y_soil = apply_derivative(y_soil, order=current_derivative, dt=frame_minutes)
            soil_curves.append(y_soil)
        
        # Organize soil curves by distance
        # Based on generate_soil_coords: [+dist1, -dist1, +dist2, -dist2] = [+4mm, -4mm, +2mm, -2mm]
        distance_curves = {
            -2: soil_curves[1],   # -4mm (second element: -dist1 where dist1=8, so -4mm)
            -1: soil_curves[3],   # -2mm (fourth element: -dist2 where dist2=4, so -2mm)
            0: y_root,            # root
            1: soil_curves[2],    # +2mm (third element: +dist2 where dist2=4, so +2mm)  
            2: soil_curves[0]     # +4mm (first element: +dist1 where dist1=8, so +4mm)
        }
        
        # Find peak times for each distance
        section_peak_times = []
        for dist_idx, dist in enumerate(distances):
            curve = distance_curves[dist]
            x_curve = x[window_size-1:] if current_smoothing == "Moving Average" else x
            x_curve_time = x_curve * frame_minutes
            
            # Find peak time
            peak_val, peak_time = find_peak_value(curve, x_curve_time)
            section_peak_times.append(peak_time if peak_time is not None else 0)
            
            # Add to global lists for plotting
            all_peak_times.append(peak_time if peak_time is not None else 0)
            colors_list.append(distance_colors[dist_idx])
            section_labels.append(f"{root_name}\n{distance_labels[dist_idx]}")
    
    # Create grouped bar chart
    x_positions = []
    bar_width = 0.15
    group_width = len(distances) * bar_width
    group_spacing = 0.3
    
    for section_idx in range(len(root_names)):
        group_start = section_idx * (group_width + group_spacing)
        for dist_idx in range(len(distances)):
            x_pos = group_start + dist_idx * bar_width
            x_positions.append(x_pos)
    
    # Create bars
    bars = ax.bar(x_positions, all_peak_times, width=bar_width, color=colors_list, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar, peak_time in zip(bars, all_peak_times):
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Customize the plot
    deriv_text = {0: "Raw Intensity", 1: "1st Derivative", 2: "2nd Derivative"}[current_derivative]
    ax.set_title(f'Peak Times Analysis - {current_dir} Direction\nSmoothing: {current_smoothing} - {deriv_text}', fontsize=14, fontweight='bold')
    ax.set_ylabel('Peak Time (minutes)', fontsize=12)
    ax.set_xlabel('Root Sections and Distances', fontsize=12)
    
    # Set x-axis labels and ticks
    group_centers = []
    for section_idx in range(len(root_names)):
        group_start = section_idx * (group_width + group_spacing)
        group_center = group_start + (group_width - bar_width) / 2
        group_centers.append(group_center)
    
    ax.set_xticks(group_centers)
    ax.set_xticklabels(root_names, fontsize=12, fontweight='bold')
    
    # Add distance labels below x-axis
    for section_idx in range(len(root_names)):
        group_start = section_idx * (group_width + group_spacing)
        for dist_idx, dist_label in enumerate(distance_labels):
            x_pos = group_start + dist_idx * bar_width
            # Position labels below the x-axis
            ax.text(x_pos, -max(all_peak_times) * 0.08, dist_label, 
                   ha='center', va='top', fontsize=9, rotation=45)
    
    # Add legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=distance_colors[i], alpha=0.8, edgecolor='black') 
                      for i in range(len(distance_labels))]
    ax.legend(legend_elements, distance_labels, loc='upper right', title='Distance from Root')
    
    # Grid and formatting
    ax.grid(True, alpha=0.3, axis='y')
    max_peak_time = max(all_peak_times) if all_peak_times else 100
    ax.set_ylim(-max_peak_time * 0.15, max_peak_time * 1.15)  # Adjusted to make room for labels
    
    # Add section dividers
    for i in range(1, len(root_names)):
        divider_x = i * (group_width + group_spacing) - group_spacing/2
        ax.axvline(x=divider_x, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    
    # Add control buttons at the bottom
    buttons = []
    
    # Direction buttons
    ax_x = plt.axes([0.1, 0.02, 0.06, 0.05])
    ax_y = plt.axes([0.15, 0.02, 0.06, 0.05])
    btn_x = Button(ax_x, 'X')
    btn_y = Button(ax_y, 'Y')
    btn_x.on_clicked(lambda _: update_dir_peaks("X"))
    btn_y.on_clicked(lambda _: update_dir_peaks("Y"))
    buttons.extend([btn_x, btn_y])
    
    # Derivative buttons
    ax_d0 = plt.axes([0.25, 0.02, 0.08, 0.05])
    ax_d1 = plt.axes([0.32, 0.02, 0.08, 0.05])
    ax_d2 = plt.axes([0.39, 0.02, 0.08, 0.05])
    btn_d0 = Button(ax_d0, 'Raw')
    btn_d1 = Button(ax_d1, '1st Der')
    btn_d2 = Button(ax_d2, '2nd Der')
    btn_d0.on_clicked(lambda _: update_derivative_peaks(0))
    btn_d1.on_clicked(lambda _: update_derivative_peaks(1))
    btn_d2.on_clicked(lambda _: update_derivative_peaks(2))
    buttons.extend([btn_d0, btn_d1, btn_d2])
    
    # Smoothing buttons
    for i, mode in enumerate(smoothing_modes):
        ax_smooth = plt.axes([0.5 + i*0.08, 0.02, 0.07, 0.05])
        btn_smooth = Button(ax_smooth, mode)
        btn_smooth.on_clicked(lambda _, m=mode: update_smoothing_peaks(m))
        buttons.append(btn_smooth)
    
    # Back to regular plot button
    ax_back = plt.axes([0.85, 0.02, 0.1, 0.05])
    btn_back = Button(ax_back, 'Regular Plot')
    btn_back.on_clicked(lambda _: plot())
    buttons.append(btn_back)
    
    plt.show()

def plot():
    global soil_coords, soil_labels, soil_dirs
    root_coord = root_positions[current_root]
    soil_coords, soil_dirs, soil_labels = generate_soil_coords(root_coord, dir=current_dir)

    root_curve, files = extract_voxel_curve(tiff_folder, root_coord, "Root", "X")
    soil_curves = [
        extract_voxel_curve(tiff_folder, soil_coords[i], "Soil", soil_dirs[i])[0]
        for i in range(len(soil_coords))
    ]

    x = np.arange(len(files))
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.35)

    y_root = apply_smoothing(root_curve)
    dt = frame_minutes
    y_root = apply_derivative(y_root, order=current_derivative, dt=dt)
    x_plot = x[window_size-1:] if current_smoothing == "Moving Average" else x

    auc_root = np.trapz(y_root, x_plot * dt)
    ax.plot(x_plot, y_root, label=f'Root {current_root} (AUC={auc_root:.1f})', lw=2, color='blue')

    for y_soil, label in zip(soil_curves, soil_labels):
        y = apply_smoothing(y_soil)
        x_soil = x[window_size-1:] if current_smoothing == "Moving Average" else x
        y = apply_derivative(y, order=current_derivative, dt=dt)
        auc = np.trapz(y, x_soil * dt) if current_derivative == 0 else None
        plot_label = f'{label} (AUC={auc:.1f})' if auc else label
        ax.plot(x_soil, y, label=plot_label, linestyle='--')

    deriv_text = {0: "None", 1: "1st", 2: "2nd"}[current_derivative]
    ax.set_title(f"ROI Curves {current_root} {current_dir} - Smoothing: {current_smoothing} - Derivative: {deriv_text}")
    ax.set_xlabel("Time Frame (min)")
    ax.set_ylabel(["Intensity", "Intensity change (per min)", "Intensity acceleration (per min^2)"][current_derivative])
    ax.set_xticks(x)
    ax.set_xlim(3, 25)
    ax.set_xticklabels([f"{i*frame_minutes}" for i in x], rotation=90)
    ax.grid(True)
    ax.legend()

    buttons = []
    def make_button(pos, label, callback):
        ax_btn = plt.axes(pos)
        btn = Button(ax_btn, label)
        btn.on_clicked(callback)
        buttons.append(btn)

    # Reorganized button layout
    # Row 1: Root selection
    make_button([0.1, 0.20, 0.08, 0.04], 'Top', lambda _: update_root("Top"))
    make_button([0.19, 0.20, 0.08, 0.04], 'Middle', lambda _: update_root("Middle"))
    make_button([0.28, 0.20, 0.08, 0.04], 'Bottom', lambda _: update_root("Bottom"))

    # Row 2: Direction
    make_button([0.1, 0.15, 0.05, 0.04], 'X', lambda _: update_dir("X"))
    make_button([0.16, 0.15, 0.05, 0.04], 'Y', lambda _: update_dir("Y"))

    # Row 3: Derivatives
    make_button([0.1, 0.10, 0.06, 0.04], 'Raw', lambda _: update_derivative(0))
    make_button([0.17, 0.10, 0.06, 0.04], '1st Der', lambda _: update_derivative(1))
    make_button([0.24, 0.10, 0.06, 0.04], '2nd Der', lambda _: update_derivative(2))

    # Row 4: Smoothing
    for i, mode in enumerate(smoothing_modes):
        make_button([0.1 + i*0.08, 0.06, 0.07, 0.04], mode, lambda _, m=mode: update_smoothing(m))

    # Row 5: Peak Analysis
    make_button([0.1, 0.02, 0.15, 0.04], 'Plot Peaks', lambda _: plot_peaks())

    fig.canvas.draw()
    plt.show()

def update_root(pos):
    global current_root
    current_root = pos
    plt.close()
    plot()

def update_dir(direction):
    global current_dir
    current_dir = direction
    plt.close()
    plot()

def update_smoothing(mode):
    global current_smoothing
    current_smoothing = mode
    plt.close()
    plot()

def update_derivative(derivative):
    global current_derivative
    current_derivative = int(derivative)
    plt.close()
    plot()

# Peak plot update functions
def update_dir_peaks(direction):
    global current_dir
    current_dir = direction
    plt.close()
    plot_peaks()

def update_smoothing_peaks(mode):
    global current_smoothing
    current_smoothing = mode
    plt.close()
    plot_peaks()

def update_derivative_peaks(derivative):
    global current_derivative
    current_derivative = int(derivative)
    plt.close()
    plot_peaks()

if __name__ == "__main__":
    plot()

