import tifffile
import napari
import numpy as np
import tkinter as tk
from tkinter import filedialog
import time

def load_zstack_tiff(tiff_path):
    """Loads a 3D Z-stack TIFF file into a NumPy array."""
    tiff_data = tifffile.imread(tiff_path)
    
    if tiff_data.ndim != 3:
        raise ValueError(f"Expected 3D TIFF stack, but got shape {tiff_data.shape}")
    
    return tiff_data

def main():
    # Use Tkinter to open file dialog
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    tiff_path = filedialog.askopenfilename(
        title="Select a TIFF Z-stack file",
        filetypes=[("TIFF files", "*.tif *.tiff")]
    )

    if not tiff_path:
        print("No file selected. Exiting.")
        return

    volume = load_zstack_tiff(tiff_path)

    # Open in Napari 3D viewer
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(volume, name='Z-stack', rendering='mip', colormap='gray')
    
    napari.run()

if __name__ == "__main__":
    main()