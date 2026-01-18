import numpy as np
import parallelproj
import nibabel as nib
import os
import shutil
import cupy as cp
import tifffile
xp = cp

# === CONFIG ===

DATE = "07162025" #date of experiment (MMDDYYYY) I personally used this to organize files
TIME_BINS = 3 #time per frame in minutes

INPUT_LM_FOLDER = rf"C:/Users/Krisps/PHYTOPet/LM_Files/07162025/Dyn_3min_lm_2/"
OUTPUT_FOLDER = f"C:/Users/Krisps/PHYTOPet/img/{DATE}_pproj_{TIME_BINS}min_65mm"
ITERATIONS = 40
SUBSETS = 1
IMAGE_SHAPE = (300, 300, 100)
VOXEL_SIZES = (0.5, 0.5, 0.5) 

cont_magnitude = 5e-5
num_TOF_bins = 351
TOF_bin_width = 1
sigma_TOF = 31.85
radius_mm = 65
iterations_of_interest = [40] #iterations we are interested in

use_tof = False
use_att = False
process_nii_to_tiff = True
sensitivity = 0.5
threshold = 0

#Creates the new directories: 
for iteration in iterations_of_interest:
    file_output_folder = f"{OUTPUT_FOLDER}_{iteration:03d}itr/"

    if os.path.exists(file_output_folder):
        shutil.rmtree(file_output_folder)  # deletes the directory and all its contents
    os.makedirs(file_output_folder, exist_ok=True)         


if use_att:
    att_nifti = nib.load(r"E:\LM\registered_att_img_water.nii").get_fdata().astype(np.float32)
    att_img = xp.asarray(att_nifti)
ref_affine = nib.load(rf"C:\Users\Krisps\PHYTOPet\config_files\res_attenuation_map_ct.nii").affine
#efficiency_per_event = np.memmap(r"E:\LM\event_efficiency.npy", dtype=np.float32, mode='r')
#scatter_per_event = np.load(r"E:\LM\res_scatter_per_event_11x.npy").astype(np.float32)
#scatter_per_event[:] = cont_magnitude
# === FORMAT LISTMODE FILE ===
print("Loading and processing listmode file...")

# GAUSS BLUR
TOR_FWHM = 2
## image based resolution model
res_model = parallelproj.GaussianFilterOperator(
    IMAGE_SHAPE, sigma=TOR_FWHM / (2.35 * np.array(VOXEL_SIZES))
)


def make_circular_mask(shape, VOXEL_SIZES, radius_mm):
    Nx, Ny, Nz = shape
    dx, dy, dz = VOXEL_SIZES

    # Create grid of physical coordinates centered at 0
    x = (xp.arange(Nx) - Nx // 2) * dx
    y = (xp.arange(Ny) - Ny // 2) * dy
    z = (xp.arange(Nz) - Nz // 2) * dz

    xx, yy, zz = xp.meshgrid(x, y, z, indexing='ij')

    # Compute radial distance from center (only xy-plane for cylindrical mask)
    r = xp.sqrt(xx**2 + yy**2)

    # Create binary mask: 1 inside detector, 0 outside
    mask = (r <= radius_mm).astype(xp.float32)
    return mask

mask = make_circular_mask(IMAGE_SHAPE, VOXEL_SIZES, radius_mm)


def create_circular_mask(length, width, radius):
    Y, X = np.ogrid[:length, :width]
    cx, cy = width // 2, length // 2
    dist_from_center = np.sqrt((X-cx)**2 + (Y-cy)**2)
    newMask = dist_from_center <= radius
    return newMask.T

newMask = create_circular_mask(IMAGE_SHAPE[0], IMAGE_SHAPE[1], radius_mm)


def process_frame(nii_data):
    
    z_slices_stack = []

    for i in range(IMAGE_SHAPE[2]):

        z_slice = nii_data[:,:, i]
        z_slice = z_slice.T
        z_slice = z_slice * sensitivity
        z_slice = np.clip(z_slice, 0, None)

        if threshold != 0:
            z_slice = np.where(z_slice >= threshold, z_slice, 0)
        
        z_slice *= newMask

        z_slices_stack.append(np.flipud(z_slice))
    
    z_slices_stack = np.array(z_slices_stack) # shape

    #crop the slices 
    midpoint = IMAGE_SHAPE[0] // 2
    cropped_z_stack = z_slices_stack[:, midpoint-radius_mm:midpoint+radius_mm, midpoint-radius_mm:midpoint+radius_mm]

    #normalize the slices
    vmin = 0
    vmax = 255

    cropped_z_stack = cropped_z_stack.astype(np.float32)
    cropped_z_stack = np.clip(cropped_z_stack, vmin, vmax)

    if (vmax-vmin != 255):
        cropped_z_stack = (cropped_z_stack - vmin) / (vmax - vmin)
        cropped_z_stack *= 255.0

    cropped_z_stack = cropped_z_stack.astype(np.uint8)

    return cropped_z_stack



# NORMALIZATION
normalization_weights = xp.asarray(np.load(r"C:\Users\Krisps\PHYTOPet\config_files\CylinderTest_parallelproj2(1).npy"))
# normalization_weights = xp.asarray(np.load(r"E:\LM\res_norm_for_registration.npy"))
epsilon = xp.array(1e-4, dtype=xp.float32)       
norm = normalization_weights.astype(xp.float32)

# A^T n / (max A^T n) + epsilon
norm_max = xp.max(norm)
norm = norm / (norm_max + xp.finfo(norm.dtype).tiny)
norm = norm + epsilon
normalization_weights=norm

def lm_em_update(x_cur, op, s):
    ybar = op(x_cur) + s
    # Print stats about op(x_cur)
    # op_x_cur = ybar - s
    # op_x_cur_cpu = op_x_cur.get() if isinstance(op_x_cur, cp.ndarray) else op_x_cur
    # print(f"op(x_cur): min={op_x_cur_cpu.min():.2e}, max={op_x_cur_cpu.max():.2e}, mean={op_x_cur_cpu.mean():.2e}")
    # sensitivity_scale = 0.4
    return x_cur * op.adjoint(1 / ybar) / (normalization_weights)


# === CHUNKED PROCESSING ===


for file in os.listdir(INPUT_LM_FOLDER):

    img = xp.full(IMAGE_SHAPE, 1e-3, dtype=np.float32)  # Use ones for stability
    img = mask * img


    chunk_size = int(1e8)
    num_floats_per_event = 10
    float_size = 4  # bytes
    event_size_bytes = num_floats_per_event * float_size
    file_size = os.path.getsize(INPUT_LM_FOLDER + file)
    num_events_total = file_size // event_size_bytes
    events_per_subset = num_events_total // SUBSETS
    print(f"Total events: {num_events_total:,}")
    print(f"Events per subset (chunk): {events_per_subset:,}")

    with open(INPUT_LM_FOLDER + file, 'rb') as f:
        for iter in range(ITERATIONS):
            print(f"\n=== Iteration {iter+1}/{ITERATIONS} ===")

            f.seek(0)
            for subset_idx in range(SUBSETS):
                print(f"Subset {subset_idx+1}/{SUBSETS}")

                buffer = np.fromfile(f, dtype=np.float32, count=events_per_subset * num_floats_per_event)
                if buffer.size == 0:
                    break

                buffer = buffer.reshape(-1, num_floats_per_event)
                start_idx = subset_idx * events_per_subset
                end_idx = start_idx + len(buffer)

                start_coord = xp.asarray(buffer[:, [0, 1, 2]])
                end_coord   = xp.asarray(buffer[:, [5, 6, 7]])
                
                if use_tof:
                    delta_time = xp.asarray(buffer[:, 8]) - xp.asarray(buffer[:, 3])  # deltaT

                    c_mm_per_ps = 0.299792  # mm/ps
                    delta_distance_mm = 0.5 * c_mm_per_ps * delta_time  # convert ps → mm
                    half_bins = num_TOF_bins // 2
                    tof_bin_indices = xp.round(delta_distance_mm / TOF_bin_width).astype(np.int32)
                    tof_bin_indices = xp.clip(tof_bin_indices, -half_bins, half_bins)
                    tof_bin_indices = xp.asarray(tof_bin_indices)  # Move to GPU

                # === FIXED INDENTATION HERE ===
                num_events_in_chunk = buffer.shape[0]
                scatter_subset = xp.full(num_events_in_chunk, cont_magnitude, dtype=xp.float32)

                proj = parallelproj.ListmodePETProjector(
                    start_coord, end_coord, IMAGE_SHAPE, VOXEL_SIZES
                )

                if use_tof:
                    proj.tof_parameters = parallelproj.TOFParameters(
                        num_tofbins=num_TOF_bins,
                        tofbin_width=TOF_bin_width,
                        sigma_tof=sigma_TOF
                    )
                    proj.event_tofbins = tof_bin_indices
                    proj.tof = True

                if use_att:
                    subset_att_list = xp.exp(-proj(att_img))
                    subset_lm_att_op = parallelproj.ElementwiseMultiplicationOperator(subset_att_list)
                    op = parallelproj.CompositeLinearOperator((subset_lm_att_op, proj, res_model))
                else:
                    op = parallelproj.CompositeLinearOperator((proj, res_model))


                img = lm_em_update(img, op, scatter_subset)


            if iter+1 in iterations_of_interest:
                
                print(f"    saving {file} at iteration {iter+1}")
                img_np = img.get()
                img_np = np.flip(img_np,axis=1)
                img_np = np.flip(img_np,axis=2)
                
                file_output_folder = f"{OUTPUT_FOLDER}_{iter+1:03d}itr/"

                if process_nii_to_tiff:
                    
                    #process nii to tiff
                    file_name = f"{file[:-3]}_{iter+1:03d}itr.tiff"

                    processed_tiff_frame = process_frame(img_np)

                    tifffile.imwrite(file_output_folder + file_name, processed_tiff_frame)
                    print(f"exported TIFF file to {file_output_folder + file_name}")

                else:
                    
                    #save nii file
                    file_name = f"{file[:-3]}_{iter+1:03d}itr.nii"
                    
                    midpoint = 150

                    img_np = img_np[midpoint-radius_mm:midpoint+radius_mm,midpoint-radius_mm:midpoint+radius_mm,:]

                    print(img_np.shape)

                    nib.save(
                        nib.Nifti1Image(img_np, affine=ref_affine),
                        os.path.join(file_output_folder, file_name)
                    )
                    print(f"exported NII file to {file_output_folder + file_name}")

print("OSEM Reconstruction complete!")

# # Image Generation 
# img_np = np.asarray(img)
#img_np = img.get()

# img_np = np.flip(img_np, axis=0)  # Flip along the x-axis, xflip is this commented out
#img_np = np.flip(img_np, axis=1)  # Flip along the y-axis
#img_np = np.flip(img_np, axis=2)  # Flip along the z-axispCRC_norm_gpu_1_noAtt.nii.gz")
