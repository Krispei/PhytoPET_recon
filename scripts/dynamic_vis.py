import numpy as np
import os
import nibabel as nib
import napari
from tkinter import Tk, filedialog
from natsort import natsorted

def choose_folder():
    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title="Select Folder of NIfTI Files")
    if not folder:
        raise FileNotFoundError("No folder selected.")
    return folder

def load_4d_volume(folder):
    # List NIfTI files sorted by name (frame order)
    # Checks for both .nii and .nii.gz
    nii_files = [os.path.join(folder, f) for f in os.listdir(folder)
                 if f.lower().endswith((".nii", ".nii.gz"))]
    nii_files = natsorted(nii_files)

    if not nii_files:
        raise FileNotFoundError("No NIfTI files found in folder.")

    print(f"Loading {len(nii_files)} NIfTI volumes...")

    # Load each 3D stack and collect as a 4D array
    # nib.load(f).get_fdata() returns the image data as a numpy array (usually float64)
    frames = [nib.load(f).get_fdata() for f in nii_files]
    
    # Ensure consistent shape
    shape0 = frames[0].shape
    for f in frames:
        if f.shape != shape0:
            raise ValueError(f"All NIfTI files must have the same shape. Found mismatch: {f.shape} vs {shape0}")
    
    # Stack frames along the new time axis (Axis 0)
    # Resulting Shape: (Time, X, Y, Z) - Note that NIfTI usually loads as (X, Y, Z)
    volume_4d = np.stack(frames, axis=0)
    return volume_4d

def main():
    try:
        folder = choose_folder()
        volume_4d = load_4d_volume(folder)

        print(f"Data loaded successfully. Shape: {volume_4d.shape}")

        viewer = napari.Viewer()
        
        # Determine contrast limits based on data range (NIfTI is often not 0-255)
        c_min, c_max = np.min(volume_4d), np.max(volume_4d)

        layer = viewer.add_image(
            volume_4d,
            name="Time Series NIfTI",
            colormap='gray',
            contrast_limits=[c_min, c_max],
            rendering='mip',
            blending='translucent',
            opacity=1.0
        )
        
        layer.interpolation = 'nearest'
        
        # Adjusting viewer settings
        # Note: NIfTI data is usually (X, Y, Z). After stacking, we have (Time, X, Y, Z).
        viewer.dims.ndisplay = 3  # Set to 3D view mode
        
        # NIfTI axes are often distinct from TIFF. We label them generically here.
        viewer.dims.axis_labels = ['time', 'x', 'y', 'z']

        napari.run()
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
