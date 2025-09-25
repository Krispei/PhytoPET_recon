import numpy as np
import nibabel as nib

# Inputs
Directory = 'SummerPHYTOPET/'
Project = 'james/'
Vox_folder = ''

file = "Hoffman_PET_normAll_1_tof_itr15.vox"

vox_path = Directory + Project + Vox_folder + file
nifti_path = Directory + Project + file[:-4] + '.nii'
# Original data shape in file (Z, Y, X)
original_shape = (350, 350, 350)
dtype = np.float32

# Voxel sizes in mm (you can change these to your actual voxel spacing)
voxel_size_x = 1
voxel_size_y = 1
voxel_size_z = 1

# Step 1: Load and reshape raw data
vox_data = np.fromfile(vox_path, dtype=dtype).reshape(original_shape)

# Step 2: Transpose from (Z, Y, X) → (X, Y, Z)
vox_data = np.transpose(vox_data, (2, 1, 0))  # Now shape is (300, 300, 100)

# Step 3: Set voxel size affine matrix for (X, Y, Z)
affine = np.array([
    [voxel_size_x, 0, 0, 0],
    [0, voxel_size_y, 0, 0],
    [0, 0, voxel_size_z, 0],
    [0, 0, 0, 1]
])

# Step 4: Create NIfTI and save
nifti_img = nib.Nifti1Image(vox_data, affine)
nib.save(nifti_img, nifti_path)

print(f"Saved NIfTI with shape {vox_data.shape} to: {nifti_path}")