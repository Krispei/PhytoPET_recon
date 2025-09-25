import os
import tifffile as tiff
import numpy as np

# === Configuration ===
input_folder = "SummerPHYTOPET/RSM.2025.07.21.21.34"
output_folder = "SummerPHYTOPET/RSM.2025.07.21.21.34_summed"
flatten_method = "sum"  # Options: "mean", "max", "sum"

os.makedirs(output_folder, exist_ok=True)

# === Processing Function ===
def flatten_stack(stack, method="mean"):
    if method == "mean":
        return np.mean(stack, axis=0)
    elif method == "max":
        return np.max(stack, axis=0)
    elif method == "sum":
        return np.sum(stack, axis=0)
    else:
        raise ValueError(f"Unknown flattening method: {method}")

# === Main Loop ===
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".tif", ".tiff")):
        input_path = os.path.join(input_folder, filename)
        print(f"Processing {filename}...")

        # Load 3D image (X, Y, Z)
        stack = tiff.imread(input_path)  # shape: (X, Y, Z)

        # Flatten along Z-axis (axis=2)
        flat_image = flatten_stack(stack, method=flatten_method)  # shape: (X, Y)

        # Save as float32 or uint16 depending on source
        if np.issubdtype(flat_image.dtype, np.floating):
            flat_image = flat_image.astype(np.float32)
        else:
            flat_image = flat_image.astype(np.uint16)

        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_flattened.tif")
        tiff.imwrite(output_path, flat_image)
        print(f"Saved to {output_path}")