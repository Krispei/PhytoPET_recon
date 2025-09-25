import numpy as np
import parallelproj
import nibabel as nib
import os
import cupy as cp
xp = cp

input_lm_file = r"G:\Shared drives\PhytoPET_Nasir_ Dr_Lee\Wonu\CylinderTest.lm"
output_name = r"G:\Shared drives\PhytoPET_Nasir_ Dr_Lee\Wonu\CylinderTest_parallelproj.npy"
radius_mm = 150
shape = (300, 300, 100)
voxel_size = (0.5, 0.5, 0.5)
# shape = (513, 513, 398)
# voxel_size = (0.48606, 0.48606, 0.48606) #mm
num_TOF_bins = 351
TOF_bin_width = 1
sigma_TOF = 31.85
use_tof = False

def make_circular_mask(shape, voxel_size, radius_mm):
    Nx, Ny, Nz = shape
    dx, dy, dz = voxel_size

    # Create grid of physical coordinates centered at 0
    x = (np.arange(Nx) - Nx // 2) * dx
    y = (np.arange(Ny) - Ny // 2) * dy
    z = (np.arange(Nz) - Nz // 2) * dz

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

    # Compute radial distance from center (only xy-plane for cylindrical mask)
    r = np.sqrt(xx**2 + yy**2)

    # Create binary mask: 1 inside detector, 0 outside
    mask = (r <= radius_mm).astype(np.float32)

    return mask

mask = make_circular_mask(shape, voxel_size, radius_mm)

print("Loading and processing listmode file...")

chunk_size = int(1e8)  # number of events per chunk (adjust based on available RAM)
event_size_bytes = 10 * 4  # 10 float32s per event
file_size_bytes = os.path.getsize(input_lm_file)
num_events_total = file_size_bytes // event_size_bytes
print(f"Total events: {num_events_total}")


att_nifti = nib.load(r"E:\LM\registered_att_img_water.nii").get_fdata().astype(np.float32)
att_img = np.asarray(att_nifti)

norm_image = np.zeros(shape, dtype=np.float32)  # initialize final result
TOR_FWHM = 2
res_model = parallelproj.GaussianFilterOperator(
    shape, sigma=TOR_FWHM / (2.35 * np.array(voxel_size))
)

with open(input_lm_file, "rb") as f:
    for chunk_start in range(0, num_events_total, chunk_size):
        num_to_read = min(chunk_size, num_events_total - chunk_start)
        data = np.fromfile(f, dtype=np.float32, count=num_to_read * 10).reshape((-1, 10))
        start_coord = data[:, [0, 1, 2]]
        end_coord = data[:, [5, 6, 7]]
        delta_distance_mm = data[:, 3]
        # c_mm_per_ps = 0.299792  # mm/ps
        # delta_distance_mm /= c_mm_per_ps
        # delta_distance_mm /= 1.5625
        # delta_distance_mm -= 512
        # delta_distance_mm *= 1.5625
        # delta_distance_mm *= c_mm_per_ps
        half_bins = num_TOF_bins // 2
        tof_bin_indices = np.round(delta_distance_mm / TOF_bin_width).astype(np.int32)
        tof_bin_indices = np.clip(tof_bin_indices, -half_bins, half_bins)

        listmode_data = np.ones(len(start_coord), dtype=np.float32)

        projector = parallelproj.ListmodePETProjector(start_coord, end_coord, shape, voxel_size)
        subset_att_list = np.exp(-projector(att_img))
        subset_lm_att_op = parallelproj.ElementwiseMultiplicationOperator(subset_att_list)
        if use_tof:
            projector.tof_parameters = parallelproj.TOFParameters(
                num_tofbins=num_TOF_bins,
                tofbin_width=TOF_bin_width,
                sigma_tof=sigma_TOF
            )
            projector.event_tofbins = tof_bin_indices
            projector.tof = True
        norm_proj = parallelproj.CompositeLinearOperator((projector, res_model)) #subset_lm_att_op, 
        partial_norm = norm_proj.adjoint(listmode_data)
        norm_image += partial_norm  # accumulate partial result

        print(f"Processed events {chunk_start} to {chunk_start + num_to_read}")

# Post-process and save
epsilon = 1e-6
norm_image *= mask
norm_image[np.isnan(norm_image)] = epsilon
norm_image[np.isinf(norm_image)] = epsilon
norm_image = np.clip(norm_image, a_min=epsilon, a_max=None)
np.save(output_name, norm_image)
nib.save(nib.Nifti1Image(norm_image, affine=np.eye(4)), r"E:\LM\res_norm_img_att_water_registered.nii")

