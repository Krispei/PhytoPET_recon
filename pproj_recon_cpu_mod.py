import os, json, time, glob
import numpy as np
import nibabel as nib
import parallelproj

xp = np  

# ---------- Utility ----------
def save_xp_as_nifti(xp_arr, voxel_size, out_path):

    if hasattr(xp_arr, 'get'):
        arr_cpu = xp_arr.get()
    else:
        arr_cpu = xp_arr
    
    affine = np.diag(tuple(voxel_size) + (1.0,))
    nib.save(nib.Nifti1Image(arr_cpu, affine=affine), out_path)

def make_circular_mask(shape, voxel_size, radius_mm):
    Nx, Ny, Nz = shape
    dx, dy, dz = voxel_size
    x = (xp.arange(Nx) - Nx // 2) * dx
    y = (xp.arange(Ny) - Ny // 2) * dy
    z = (xp.arange(Nz) - Nz // 2) * dz
    xx, yy, zz = xp.meshgrid(x, y, z, indexing='ij')
    r = xp.sqrt(xx**2 + yy**2)
    return (r <= radius_mm).astype(xp.float32)

def build_res_model(image_shape, voxel_size, tor_fwhm_mm):
    return parallelproj.GaussianFilterOperator(
        image_shape, sigma = tor_fwhm_mm / (2.35 * xp.asarray(voxel_size))
    )

def event_reader(file_obj, n_events, floats_per_event=10):
    buf = np.fromfile(file_obj, dtype=np.float32, count=n_events * floats_per_event)
    if buf.size == 0:
        return None
    return buf.reshape(-1, floats_per_event)

def compute_tof_bins(delta_distance_mm, num_tof_bins, tof_bin_width_mm):
    half = num_tof_bins // 2
    tof_bins = xp.rint(delta_distance_mm / float(tof_bin_width_mm)).astype(xp.int32)
    return xp.clip(tof_bins, -half, half)

# ---------- Normalization Loader ----------
def load_normalization_npy(cfg):
    """
    Loads a pre-generated normalization map from a .npy file.
    Returns the ready-to-use normalization weights array.
    """
    npy_path = cfg.get("norm_npy_path")
    if not npy_path or not os.path.exists(npy_path):
        raise FileNotFoundError(f"Normalization .npy file not found at: {npy_path}")
    
    print(f"[Norm] Loading pre-generated map: {npy_path}")
    norm_img = np.load(npy_path).astype(xp.float32)
    
    # Verify shape matches config
    expected_shape = tuple(cfg["image_shape"])
    if norm_img.shape != expected_shape:
        raise ValueError(f"Norm shape {norm_img.shape} does not match config {expected_shape}")

    # --- Pre-process weights (once) ---
    # Normalize to max=1 and add small smoothing factor to avoid division by zero
    norm_max = xp.max(norm_img)
    if norm_max == 0:
        raise ValueError("Normalization map is all zeros!")
        
    norm_img = norm_img / (norm_max + xp.finfo(norm_img.dtype).tiny)
    normalization_weights = norm_img + xp.array(1e-4, dtype=xp.float32)
    
    return normalization_weights


# ---------- OSEM LM reconstruction (Single Frame) ----------
def run_recon_single_file(cfg, lm_path, norm_weights, frame_name):
    out_dir = cfg["output_dir"]
    image_shape     = tuple(cfg["image_shape"])
    voxel_size      = tuple(cfg["voxel_size"])
    n_iter          = int(cfg.get("num_iterations", 5))
    n_subsets       = int(cfg.get("num_subsets", 4))
    cont_mag        = float(cfg.get("cont_magnitude", 1e-5))

    # Physics parameters
    use_tof         = bool(cfg.get("use_tof_recon", cfg.get("use_tof", True)))
    num_tof_bins    = int(cfg.get("num_TOF_bins", 401))
    tof_bin_width   = float(cfg.get("TOF_bin_width", 1.0))
    fwhm_tof        = float(cfg.get("fwhm_tof", 37.5))
    sigma_tof       = xp.asarray([fwhm_tof / 2.35], dtype=xp.float32)
    tor_fwhm        = float(cfg.get("TOR_FWHM", 3.0))
    
    use_mask        = bool(cfg.get("use_mask", False))
    mask_radius_mm  = float(cfg.get("mask_radius_mm", 140.0))
    use_att         = bool(cfg.get("use_att", False))
    
    att_img = None
    if use_att:
        # Note: If attenuation maps change per frame (e.g. motion), this needs logic to load matching file
        att_img = nib.load(cfg["att_nifti_path"]).get_fdata().astype(xp.float32)
        att_img = xp.asarray(att_img)

    # File stats
    floats_per_event= 10
    event_bytes     = floats_per_event * 4
    num_events_total= os.path.getsize(lm_path) // event_bytes
    
    if num_events_total == 0:
        print(f"[Skip] {frame_name} is empty.")
        return

    events_per_subset = num_events_total // n_subsets

    # Init image
    img = xp.ones(image_shape, dtype=xp.float32)
    if use_mask:
        img *= make_circular_mask(image_shape, voxel_size, mask_radius_mm)

    res_model = build_res_model(image_shape, voxel_size, tor_fwhm)

    def em_update(x_cur, op, s):
        # OSEM Update: img / norm * Backproject( 1 / (Forward(img) + scatter) )
        ybar = op(x_cur) + s
        # Note: norm_weights is passed in as argument
        correction = op.adjoint(1.0 / ybar)
        img = x_cur * correction / norm_weights
        return img

    print(f"  > Processing {frame_name}: {num_events_total:,} events")
    
    with open(lm_path, "rb") as f:
        for it in range(n_iter):
            f.seek(0)
            for ss in range(n_subsets):
                buf = event_reader(f, events_per_subset, floats_per_event)
                if buf is None or buf.size == 0: break

                buf_x = xp.asarray(buf)
                start_coord = buf_x[:, [0,1,2]]
                end_coord   = buf_x[:, [5,6,7]]
                delta_distance_mm = -buf_x[:, 3] / 2.0

                proj = parallelproj.ListmodePETProjector(start_coord, end_coord, image_shape, voxel_size)
                
                if use_tof:
                    tof_bins = compute_tof_bins(delta_distance_mm, num_tof_bins, tof_bin_width)
                    proj.tof_parameters = parallelproj.TOFParameters(
                        num_tofbins=num_tof_bins, tofbin_width=tof_bin_width, sigma_tof=sigma_tof)
                    proj.event_tofbins = tof_bins
                    proj.tof = True

                # Constant background (randoms/scatter approximation)
                s = xp.full(buf_x.shape[0], cont_mag, dtype=xp.float32)

                if use_att:
                    subset_att_list = xp.exp(-proj(att_img))
                    subset_lm_att_op = parallelproj.ElementwiseMultiplicationOperator(subset_att_list)
                    A = parallelproj.CompositeLinearOperator((subset_lm_att_op, proj, res_model))
                else:
                    A = parallelproj.CompositeLinearOperator((proj, res_model))

                img = em_update(img, A, s)
    
    # Save final result
    out_nii = os.path.join(out_dir, f"{cfg.get('recon_prefix','recon')}_{frame_name}.nii")
    save_xp_as_nifti(img, voxel_size, out_nii)
    print(f"  > Saved {out_nii}")


# ---------- Main ----------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="Path to params JSON")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    # 1. Load Pre-generated Normalization (Once)
    norm_weights = load_normalization_npy(cfg)

    # 2. Get Dynamic Files
    lm_folder = cfg.get("input_lm_folder")

    print(f"Processing {lm_folder}...")

    if not lm_folder or not os.path.isdir(lm_folder):
        raise ValueError("Params must contain valid 'input_lm_folder' directory.")

    pattern = cfg.get("lm_file_pattern", "*") 
    files = sorted(glob.glob(os.path.join(lm_folder, pattern)))
    files = [f for f in files if os.path.isfile(f) and not f.endswith(".json") and not f.endswith(".npy")]

    print(f"[Main] Found {len(files)} dynamic frames in {lm_folder}")

    # 3. Reconstruct Frames
    total_start = time.time()
    for i, lm_file in enumerate(files):
        frame_name = f"frame_{i:03d}"
        run_recon_single_file(cfg, lm_file, norm_weights, frame_name)

    print(f"[Main] All frames done in {(time.time() - total_start)/60:.1f} min.")

if __name__ == "__main__":
    main()