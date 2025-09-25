import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button
import tifffile

directory = 'SummerPHYTOPET/'
project = 'plant_raw_noglucose_07252025/'
input_folder = 'plant_raw_noglucose_07252025_Dyn_5min_nii/'
output_folder = 'MOVIE_Plant_raw_noglucose_07252025_5min_/'

input_dir = "SummerPHYTOPET/Plant_raw_data_St_10_40am/Plant_raw_data_St_10_40_nii_7"
output_dir = "SummerPHYTOPET/Plant_raw_data_St_10_40am/MOVIE_pproj_plant_raw_data_10_40"


class NiftiViewerWithMask:
    def __init__(self, volume, mask_radius=100, outFile="out.tiff", showGraph=True):
        self.volume = volume
        self.outFile = outFile
        self.indicies = [volume.shape[0]//2, volume.shape[1]//2, volume.shape[2]//2]
        #print(self.indicies[0])
        #self.z_index = volume.shape[2] // 2
        self.sensitivity = 0.35
        self.x_max = volume.shape[0] - 1
        self.y_max = volume.shape[1] - 1
        self.z_max = volume.shape[2] - 1
        self.mask_radius = mask_radius
        self.threshold = 27

        self.mask_arial = self.create_circular_mask(h=volume.shape[0], w=volume.shape[1], radius=mask_radius)
        self.mask_coronal = self.create_cor_sag_mask(h=volume.shape[2], w=volume.shape[0],radius=mask_radius)
        self.mask_sagittal = self.create_cor_sag_mask(h=volume.shape[2], w=volume.shape[1],radius=mask_radius)

        # Set up figure and axes
        #nrows=1, ncols=3, figsize=(12,4)
        self.fig, self.ax = plt.subplots( nrows=1, ncols=3, figsize=(12,4) )
        plt.subplots_adjust(bottom=0.35, top=0.85)  # Leave room for sliders and button

        # Show initial image
        coronal_data = self.get_scaled_masked_slice( dir=0 )
        sagittal_data = self.get_scaled_masked_slice( dir=1 )
        arial_data = self.get_scaled_masked_slice( dir=2 )
        
        #self.img = 
        self.ax[0].imshow(coronal_data, cmap="hot", origin="lower")
        self.ax[0].set_title(f"Coronal View, Slice 0 | Sensitivity: {self.sensitivity:.1f}")
        #self.z_index

        self.ax[1].imshow(sagittal_data, cmap="hot", origin="lower")
        self.ax[1].set_title(f"Sagittal View, Slice 0 | Sensitivity: {self.sensitivity:.1f}")
        #self.ax[1].imshow()

        self.ax[2].imshow(arial_data, cmap="hot", origin="lower")
        self.ax[2].set_title(f"Arial View, Slice 0 | Sensitivity: {self.sensitivity:.1f}")
        #self.ax[2].imshow()
   
        # X-slice slider
        ax_xslice = plt.axes([0.25, 0.2, 0.5, 0.03])
        self.slider_xslice = Slider(ax_xslice, "X Slice", 0, self.x_max, valinit=self.indicies[0], valstep=1)
        self.slider_xslice.on_changed(self.on_sagittal_change)

        # Y-slice slider
        ax_yslice = plt.axes([0.25, 0.15, 0.5, 0.03])
        self.slider_yslice = Slider(ax_yslice, "Y Slice", 0, self.y_max, valinit=self.indicies[1], valstep=1)
        self.slider_yslice.on_changed(self.on_coronal_change)

        # Z-slice slider
        ax_zslice = plt.axes([0.25, 0.1, 0.5, 0.03])
        self.slider_zslice = Slider(ax_zslice, "Z Slice", 0, self.z_max, valinit=self.indicies[2], valstep=1)
        self.slider_zslice.on_changed(self.on_arial_change)

        # Threshold slider
        ax_thresh = plt.axes([0.25, 0.25, 0.5, 0.03])
        self.slider_thresh = Slider(ax_thresh, "Threshold", 0.0, 100, valinit=0.0, valstep=0.1)
        self.slider_thresh.on_changed(self.on_threshold_change)

        # Sensitivity slider
        ax_sens = plt.axes([0.25, 0.3, 0.5, 0.03])
        self.slider_sens = Slider(ax_sens, "Sensitivity", 0, 10, valinit=self.sensitivity, valstep=0.05)
        self.slider_sens.on_changed(self.on_sensitivity_change)


        #Export button
        ax_button = plt.axes([0.81, 0.90, 0.1, 0.05])
        self.button_export = Button(ax_button, 'Export')
        self.button_export.on_clicked(self.on_export_click)

        
        self.update_image()

        if(showGraph):
            plt.show()

    def create_circular_mask(self, h, w, radius):
        Y, X = np.ogrid[:h, :w]
        cx, cy = w // 2, h // 2
        dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
        mask = dist_from_center <= radius
        return mask.T  # Transpose to match image orientation
    
    def create_cor_sag_mask(self, h, w, radius, coord=None):
        if coord is None:
            coord = self.indicies[0]//2
        clength = w // 2

        map_dist_from_center = np.tile(np.arange(w), (h, 1)) - clength
        dist_from_center = np.abs(coord-clength)

        if (dist_from_center > radius):
            boundary = -99
        else:
            boundary = self.mask_radius * np.cos( (np.pi / 2) * (dist_from_center / radius) )

        mask = np.abs(map_dist_from_center) <= boundary
        return mask

    #dir = 0: Coronal, 1: Sagittal, 2: Arial
    def get_scaled_masked_slice(self, coord=None, dir=0):
        if coord is None:
            coord = self.indicies[dir]
        
        slicers = [
            lambda: self.volume[:, coord, :],  # coronal
            lambda: self.volume[coord, :, :],  # sagittal
            lambda: self.volume[:, :, coord],  # arial
        ]

        masks = [self.mask_coronal, self.mask_sagittal, self.mask_arial]
    
        slice_raw = slicers[dir]()
        
        # Optional: flip only if needed
        if dir == 0:  # coronal
            slice_raw = np.rot90(slice_raw, k=1)
        elif dir == 1:  # sagittal
            slice_raw = np.flipud(slice_raw.T)
        elif dir == 2:  # axial
            slice_raw = slice_raw.T
        
        slice_scaled = np.clip(slice_raw * self.sensitivity, 0, None)
        slice_scaled = np.where(slice_scaled >= self.threshold, slice_scaled, 0)
        slice_scaled = slice_scaled * masks[dir]
        #slice_scaled = np.flipud(slice_scaled)
        return slice_scaled

    def update_image(self, update_type='all'):
        if update_type in ['all', 'coronal']:
            coronal_data = self.get_scaled_masked_slice(coord=self.indicies[0], dir=0)
            self.ax[0].images[0].set_data(coronal_data)
            self.ax[0].images[0].set_clim(0, 100)

        if update_type in ['all', 'sagittal']:
            sagittal_data = self.get_scaled_masked_slice(coord=self.indicies[1], dir=1)
            self.ax[1].images[0].set_data(sagittal_data)
            self.ax[1].images[0].set_clim(0,100)

        if update_type in ['all', 'arial']:
            arial_data = self.get_scaled_masked_slice(coord=self.indicies[2], dir=2)
            self.ax[2].images[0].set_data(arial_data)
            self.ax[2].images[0].set_clim(0, 100)

        self.fig.canvas.draw_idle()
    
    def on_sensitivity_change(self, val):
        self.sensitivity = val
        self.update_image()
    
    def on_arial_change(self, val):
        self.indicies[2] = int(val)
        self.update_image(update_type='arial')
    
    def on_coronal_change(self, val):
        #I dont know why, but the scroll is reversed on display, thus the val is reversed
        self.indicies[0] = int(val)
        self.mask_coronal = self.create_cor_sag_mask(h=self.z_max+1, w=self.x_max+1,radius=self.mask_radius, coord=int(val))
        self.update_image(update_type='coronal')

    def on_sagittal_change(self, val):
        self.indicies[1] = int(val)
        self.mask_sagittal = self.create_cor_sag_mask(h=self.z_max+1, w=self.y_max+1,radius=self.mask_radius, coord=int(val))
        self.update_image(update_type='sagittal')
    
    def on_threshold_change(self, val):
        self.threshold = val
        self.update_image()
    
    def export_slices_as_stack(self, arial_filename="Arial.tiff"):
        slices_list_arial = []
        for i in range(self.volume.shape[2]):
            slice_img = self.get_scaled_masked_slice(coord=i, dir=2)
            slices_list_arial.append(np.flipud(slice_img))
        slices_array_arial = np.array(slices_list_arial)  # shape (num_slices, H, W)
        vmin = 0
        vmax = 30
        gamma = 1

        slices_array_arial = slices_array_arial.astype(np.float32)        
        slices_array_arial = np.clip(slices_array_arial, vmin, vmax)
        slices_array_arial = (slices_array_arial - vmin) / (vmax - vmin)
        slices_array_arial *= 255.0
        slices_array_arial *= gamma

        slices_array_arial = slices_array_arial.astype(np.uint8)

        #slices_array_arial[:,148:152, 148:152] = 0

        # Save the multipage TIFF
        tifffile.imwrite(arial_filename, slices_array_arial)
        print(f"Exported {slices_array_arial.shape[0]} unnormalized slices to '{arial_filename}'")

    def on_export_click(self, event):
        
        self.sensitivity = self.slider_sens.val
        self.threshold = self.slider_thresh.val

        print(f"Exporting with sensitivity={self.sensitivity}, threshold={self.threshold}")
        self.export_slices_as_stack()

import os
import glob

nifti_paths = sorted(glob.glob(os.path.join(input_dir, "*.nii*")))

print(f"Found {len(nifti_paths)} NIfTI files.")

# Load and display the middle file for manual sensitivity/threshold setting
first_path = nifti_paths[len(nifti_paths) // 2]
print(f"\nOpen viewer to set parameters: {first_path}")
Nifti_img = nib.load(first_path)
nii_data = np.squeeze(Nifti_img.get_fdata())

# Launch viewer to pick sensitivity/threshold
viewer = NiftiViewerWithMask(nii_data, mask_radius=50)
plt.show()

# Grab sensitivity/threshold after interactive session
final_sens = viewer.sensitivity
final_thresh = viewer.threshold
print(f"\nUsing sensitivity={final_sens}, threshold={final_thresh} for all remaining files...")

# Function to process & export without launching viewer
def export_nifti_stack(volume, mask_radius, arial_filename, sensitivity, threshold):
    dummy_viewer = NiftiViewerWithMask(volume, mask_radius=mask_radius, showGraph=False)
    dummy_viewer.sensitivity = sensitivity
    dummy_viewer.threshold = threshold

    print(dummy_viewer.sensitivity)

    dummy_viewer.export_slices_as_stack(arial_filename)
    plt.close(dummy_viewer.fig)

# Loop over all NIfTI files using fixed params
for path in nifti_paths:
    print(f"\nProcessing: {path}")
    Nifti_img = nib.load(path)
    nii_data = np.squeeze(Nifti_img.get_fdata())

    base_name = os.path.splitext(os.path.basename(path))[0]
    if base_name.endswith(".nii"):
        base_name = base_name[:-4]


    #export_arial_folder = os.path.join(export_dir, "Grass_Dynamic_Cylinder_Arial_Movie")
    os.makedirs(output_dir, exist_ok=True)

    arial_out = os.path.join(output_dir, f"{base_name}.tiff")


    export_nifti_stack(
        nii_data,
        mask_radius=50,
        arial_filename=arial_out,
        sensitivity=final_sens,
        threshold=final_thresh
    )