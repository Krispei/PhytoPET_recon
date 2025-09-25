import nibabel as nib
import numpy as np

# Load the NIfTI file
nifti_path = 'SummerPHYTOPET/CylinderTest_parallelproj2(1).nii'         # Replace with your actual .nii file path
nifti_img = nib.load(nifti_path)

# Get the image data as a NumPy array
nifti_data = nifti_img.get_fdata().astype(np.float32)

# Save to .npy
output_path = 'SummerPHYTOPET/CylinderTest_parallelproj2(1).npy'      # Replace with your desired output filename
np.save(output_path, nifti_data)