import numpy as np
import nibabel as nib
import os

# Inputs
directory = 'SummerPHYTOPET/'
project = '07212025/'
folder = '07212025_Dyn_5min_vox/'

# Original data shape in file (Z, Y, X)
original_shape = (100, 300, 300)
dtype = np.float32

voxel_size_x = 0.5
voxel_size_y = 0.5
voxel_size_z = 0.5

# Step 1: Load and reshape raw data

#Set voxel size affine matrix for (X, Y, Z)
affine = np.array([
    [voxel_size_x, 0, 0, 0],
    [0, voxel_size_y, 0, 0],
    [0, 0, voxel_size_z, 0],
    [0, 0, 0, 1]
])


nifti_data_folder_name = directory + project + folder

outFolder_name = "07212025_Dyn_5min_nii/"

os.mkdir(directory + outFolder_name)

for filename in os.listdir(nifti_data_folder_name):

    vox_data = np.fromfile(nifti_data_folder_name+filename, dtype=dtype).reshape(original_shape)
    # Step 2: Transpose from (Z, Y, X) → (X, Y, Z)
    vox_data = np.transpose(vox_data, (2, 1, 0))  # Now shape is (300, 300, 100)

    nifti_img = nib.Nifti1Image(vox_data, affine)
    nib.save(nifti_img, directory+outFolder_name+f"Frame_{filename}.nii")
    print(f"Saved {filename}")

print('fin')
