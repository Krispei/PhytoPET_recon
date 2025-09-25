import numpy as np
import os
from skimage import io
import napari
from tkinter import Tk, filedialog
from natsort import natsorted

def choose_folder():
    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select Folder of 3D TIFF Stacks")
    if not folder:
        raise FileNotFoundError("No folder selected.")
    return folder

def load_4d_volume(folder):
    # List TIFF files sorted by name (frame order)
    tiff_files = [os.path.join(folder, f) for f in os.listdir(folder)
                  if f.lower().endswith((".tif", ".tiff"))]
    tiff_files = natsorted(tiff_files)

    if not tiff_files:
        raise FileNotFoundError("No TIFF files found in folder.")

    print(f"Loading {len(tiff_files)} TIFF stacks...")

    # Load each 3D stack and collect as a 4D array
    frames = [io.imread(f).astype(np.uint8) for f in tiff_files]
    
    # Ensure consistent shape
    shape0 = frames[0].shape
    for f in frames:
        if f.shape != shape0:
            raise ValueError("All TIFF stacks must have the same shape.")
    
    volume_4d = np.stack(frames, axis=0)  # shape: (T, Z, Y, X)
    return volume_4d

def main():
    folder = choose_folder()
    volume_4d = load_4d_volume(folder)

    viewer = napari.Viewer()
    layer = viewer.add_image(
        volume_4d,
        name="Time Series 3D",
        colormap='gray',
        contrast_limits=[0, 255],
        scale=[1, 1, 1],  # (T, Z, Y, X)
        rendering='mip',
        blending='translucent',
        opacity=1.0
    )
    
    layer.interpolation = 'nearest'
    viewer.dims.ndisplay = 2  # 3D mode
    viewer.dims.axis_labels = ['time', 'y', 'x']

    napari.run()    

if __name__ == "__main__":
    main()
