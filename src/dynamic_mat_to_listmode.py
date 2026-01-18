import numpy as np
import h5py
import time
import os

def getDetVectors(det_file):
    # Load all at once using float32 to match output precision
    det_vectors = np.loadtxt(det_file, dtype=np.float32, delimiter=',')
    
    # Slice to get vectors (0::3 means start at 0, step by 3)
    O_vs = det_vectors[0::3, :]
    H_vs = det_vectors[1::3, :]
    V_vs = det_vectors[2::3, :]
    
    return O_vs, H_vs, V_vs

def process_geometry_math(chunk_data, O_vs, H_vs, V_vs):
    """
    Performs the heavy geometric lifting and rotation.
    Does NOT handle time splitting. Returns the full processed batch (N x 10).
    """
    
    # 1. Extract columns
    # chunk_data columns: [x1, y1, det1, x2, y2, det2, theta]
    h1 = chunk_data[:, 0][:, np.newaxis] - 21.5
    v1 = chunk_data[:, 1][:, np.newaxis] - 21.5
    d1 = chunk_data[:, 2].astype(int) - 1 
    
    h2 = chunk_data[:, 3][:, np.newaxis] - 21.5
    v2 = chunk_data[:, 4][:, np.newaxis] - 21.5
    d2 = chunk_data[:, 5].astype(int) - 1
    
    thetas = chunk_data[:, 6]

    # 2. Vectorized Geometry Calculation
    xyz1 = O_vs[d1] + (h1 * H_vs[d1]) + (v1 * V_vs[d1])
    xyz2 = O_vs[d2] + (h2 * H_vs[d2]) + (v2 * V_vs[d2])

    # 3. Rotation (Vectorized)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # Rotate Point 1
    x1_raw, y1_raw, z1 = xyz1[:, 0], xyz1[:, 1], xyz1[:, 2]
    x1_rot = x1_raw * cos_t - y1_raw * sin_t
    y1_rot = x1_raw * sin_t + y1_raw * cos_t

    # Rotate Point 2
    x2_raw, y2_raw, z2 = xyz2[:, 0], xyz2[:, 1], xyz2[:, 2]
    x2_rot = x2_raw * cos_t - y2_raw * sin_t
    y2_rot = x2_raw * sin_t + y2_raw * cos_t

    # 4. Construct Final Array (N x 10)
    num_events = len(thetas)
    output_batch = np.zeros((num_events, 10), dtype=np.float32)
    
    output_batch[:, 0] = x1_rot
    output_batch[:, 1] = y1_rot
    output_batch[:, 2] = z1
    # Cols 3,4 are zeros
    output_batch[:, 5] = x2_rot
    output_batch[:, 6] = y2_rot
    output_batch[:, 7] = z2
    # Cols 8,9 are zeros

    return output_batch

def main():
    # --- Configuration ---
    Directory = '07162025/'
    matlab_raw_filename = 'Plant_349PM_raw.E20..nCOT.RS.mat'
    det_filename = "/Users/wonupark/Desktop/PhytoPET/scripts/NewRingD162.d.txt"
    input_file = Directory + matlab_raw_filename

    # --- DYNAMIC FRAMING SETTINGS ---
    FRAME_DURATION = 60.0  # Duration of each time bucket in seconds
    
    # !!! IMPORTANT: Check your HDF5 file for the exact name of the Time dataset !!!
    # Common names: 'Time', 'T', 'time', 'coinc_time'
    time_key = 'Time' 
    
    # Keys: h1, v1, det1, h2, v2, det2, theta, TIME
    keys_of_interest = ['x1', 'y1', 'p1', 'x2', 'y2', 'p2', 'RA', time_key]
    
    BATCH_SIZE = 100000 

    # --- Setup Output Directory ---
    base_name = os.path.splitext(matlab_raw_filename)[0]
    output_dir = os.path.join(Directory, base_name + "_dynamic_frames")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    else:
        # Optional: Clear existing files in that folder to prevent appending to old data
        print(f"Output directory exists: {output_dir}")

    print("Loading detector geometry...")
    O_vs, H_vs, V_vs = getDetVectors(det_filename)

    tic = time.perf_counter()

    # --- Main Pipeline ---
    with h5py.File(input_file, 'r') as f:
        
        # Verify keys exist
        for k in keys_of_interest:
            if k not in f:
                raise KeyError(f"Key '{k}' not found in HDF5 file. Check your 'time_key' setting.")

        # Access datasets (pointers only)
        datasets = [f[k] for k in keys_of_interest]
        total_entries = datasets[0].shape[1]
        
        print(f"Total events to process: {total_entries:,}")
        print(f"Splitting into frames of {FRAME_DURATION} seconds.")
        
        for start_idx in range(0, total_entries, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, total_entries)
            
            # 1. READ CHUNK
            # Stack into (N, 8) -> The last column is Time
            chunk_data = np.stack([d[0, start_idx:end_idx] for d in datasets], axis=1).astype(np.float32)
            
            # Separate Geometry data (cols 0-6) from Time (col 7)
            geo_data = chunk_data[:, :7]
            time_data = chunk_data[:, 7]
            
            # 2. PROCESS GEOMETRY (Vectorized for the whole chunk)
            processed_batch = process_geometry_math(geo_data, O_vs, H_vs, V_vs)
            
            # 3. DYNAMIC BUCKETING
            # Calculate which frame index each event belongs to
            # e.g., Time 65.0 // 60.0 = Frame 1
            frame_indices = (time_data // FRAME_DURATION).astype(int)
            unique_frames = np.unique(frame_indices)
            
            # Iterate through the unique frames found in this chunk and write to respective files
            for frame_idx in unique_frames:
                # Create mask for events belonging to this frame
                mask = frame_indices == frame_idx
                
                # Extract subset
                frame_subset = processed_batch[mask]
                
                # Construct filename: e.g., filename_frame_0.lm
                frame_filename = f"{base_name}_frame_{frame_idx:03d}.lm"
                full_path = os.path.join(output_dir, frame_filename)
                
                # Append to binary file
                with open(full_path, 'ab') as out_f:
                    frame_subset.tofile(out_f)

            # Feedback
            if (end_idx // BATCH_SIZE) % 5 == 0 or end_idx == total_entries:
                toc = time.perf_counter()
                elapsed_min = (toc - tic) / 60
                percent = (end_idx / total_entries) * 100
                print(f"Processed {end_idx:,}/{total_entries:,} ({percent:.1f}%) in {elapsed_min:.2f} min")

    print("Process complete.")

if __name__ == "__main__":
    main()