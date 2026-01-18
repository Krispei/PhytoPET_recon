import nibabel as nib
import numpy as np
import tifffile
import os
import glob


input_dir = "05132025/05132025_dyn_3min_pproj_40itr" #input folder path goes here
output_dir = "05132025/05132025_dyn_3min_pproj_40itr_TIFF" #output folder path goes here

sensitivity = 1
threshold = 0

volume_shape = (120,120,100)
mask_radius = 60 #pixel radius 

def create_circular_mask(length, width, radius):
    Y, X = np.ogrid[:length, :width]
    cx, cy = width // 2, length // 2
    dist_from_center = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = dist_from_center <= radius
    return mask.T 

mask = create_circular_mask(volume_shape[0], volume_shape[1], mask_radius)

nifti_frames = sorted(glob.glob(f"{input_dir}/*.nii*"))
print(f"Found {len(nifti_frames)} NIfTI files.")
print(f"\nUsing sensitivity={sensitivity}, threshold={threshold} for all files...")


#loops through every frame
for frame in nifti_frames:
    #exports the nii data from one frame
    print(f"\nProcessing: {frame}")
    Nifti_img = nib.load(frame)
    nii_data = np.squeeze(Nifti_img.get_fdata())

    #Gets the name of each file (each file corresponds to one frame)
    base_name = os.path.splitext(os.path.basename(frame))[0]
    if base_name.endswith(".nii"):
        base_name = base_name[:-4]

    #creates the output directory
    os.makedirs(output_dir, exist_ok=True)

    #creates each file output path
    frame_output_path = os.path.join(output_dir, f"{base_name}.tiff")

    
    slices_list_arial = []

    #Processes all the z slices for each frame
    for i in range(volume_shape[2]):

        z_slice = nii_data[:, :, i]  
        z_slice = z_slice.T
        z_slice = z_slice * sensitivity
        z_slice = np.clip(z_slice, 0, None)

        if threshold != 0:
            z_slice = np.where(z_slice >= threshold, z_slice, 0)
        
        z_slice[~mask] = 0

        slices_list_arial.append(np.flipud(z_slice))

    slices_array_arial = np.array(slices_list_arial)  # shape (num_slices, H, W)

    #crops the slices
    midpoint = volume_shape[0] // 2 #assumes the width = length
    cropped_slices = slices_array_arial[
                                        :,
                                        midpoint - mask_radius:midpoint + mask_radius, 
                                        midpoint - mask_radius:midpoint + mask_radius
                                        ]

    #normalization scales
    vmin = 0
    vmax = 255

    #normalizes the pixel values from 0 to 255
    cropped_slices = cropped_slices.astype(np.float32)        
    cropped_slices = np.clip(cropped_slices, vmin, vmax)
    
    if (vmax-vmin != 255):
        cropped_slices = (cropped_slices - vmin) / (vmax - vmin)
        cropped_slices *= 255.0 #scales to tiff image

    cropped_slices = cropped_slices.astype(np.uint8)


    # Save the multipage TIFF
    tifffile.imwrite(frame_output_path, cropped_slices)
    print(f"Exported to '{frame_output_path}'")

print("export finshed!")