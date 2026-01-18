"""

import numpy as np
import h5py
import scipy.io
import time
import os
import matplotlib.pyplot as plt
from itertools import islice

def save_selected_keys_to_ascii(file_path, output_file, keys_of_interest, chunk_size=1000, feedback_interval=1000):

    with h5py.File(file_path, 'r') as f:

        num_entries = f[keys_of_interest[0]].shape[1]

        num_keys = len(keys_of_interest)

        # Preload all key data into memory

        all_key_data = {key: f[key][0, :] for key in keys_of_interest}

        #print(all_key_data)

        with open(output_file, 'w') as out_file:

            # Determine the number of chunks

            num_chunks = (num_entries + chunk_size - 1) // chunk_size

            for chunk_index in range(num_chunks):

                start_idx = chunk_index * chunk_size

                end_idx = min(start_idx + chunk_size, num_entries)

                # Initialize an array to store the current chunk of data

                chunk_array = np.zeros((end_idx - start_idx, num_keys))

                # Loop over each key and populate the corresponding column in the chunk_array

                for key_idx, key in enumerate(keys_of_interest):

                    chunk_array[:, key_idx] = all_key_data[key][start_idx:end_idx].flatten()

                # Write the chunk to the output file

                np.savetxt(out_file, chunk_array, delimiter=' ', fmt='%.3f')

                # Print progress feedback

                if (chunk_index + 1) % feedback_interval == 0 or (chunk_index + 1) == num_chunks:

                    percent_complete = (chunk_index + 1) / num_chunks * 100

                    print(f"Processing chunk {chunk_index + 1:,}/{num_chunks:,} ({percent_complete:.2f}% complete)")


def getDetVectors(det_file):

    det_vectors = np.loadtxt(det_file, dtype=float, delimiter=',')

    #O_vs gets the "theta vector of each detector"
    O_vs = det_vectors[0::3,:]
    #H_vs gets the "azimuthal vector of each detec"
    H_vs = det_vectors[1::3,:]
    #Vertical, should always be like 0,0,1
    V_vs = det_vectors[2::3,:]

    return O_vs, H_vs, V_vs
    
    
def rotate_lines_about_z(lines, thetas):

    # Extract the x, y, z coordinates of both endpoints for all lines

    x1, y1, z1 = lines[:, 0], lines[:, 1], lines[:, 2]

    x2, y2, z2 = lines[:, 3], lines[:, 4], lines[:, 5]

    # Precompute cos(theta) and sin(theta) for all thetas
    
    #print(thetas*4)

    cos_theta = np.cos(thetas)

    sin_theta = np.sin(thetas)

    # Apply the rotation to the first points (x1, y1)

    x1_rot = (cos_theta * x1) + (-sin_theta * y1)

    y1_rot = (sin_theta * x1) + (cos_theta * y1)

    # Apply the rotation to the second points (x2, y2)

    x2_rot = (cos_theta * x2) + (-sin_theta * y2)

    y2_rot = (sin_theta * x2) + (cos_theta * y2)

    # Reconstruct the rotated lines, keeping z coordinates the same

    zeros = np.zeros((len(x1_rot),1))

    rotated_lines = np.column_stack([x1_rot, y1_rot, z1, zeros, zeros, x2_rot, y2_rot, z2, zeros, zeros])

    return rotated_lines

def processCoins(coin_file, det_file, output_file, xyz_shift, batch_size, N):

    # remove previous output files by the same name

    try:

        os.remove(output_file)

    except OSError:

        pass

    # initialize

    coin_counter = 0

    update_point = N*batch_size

    O_vs, H_vs, V_vs = getDetVectors(det_file)

    tic = time.perf_counter()



    with open(coin_file, 'r') as coins, open(output_file, 'ab') as output_fID:
        

        while True:
            coin_batch = np.fromstring(' '.join(islice(coins, batch_size)), sep=' ', dtype=float).reshape(-1, 7)

            #print(coin_batch)

            #print(coin_batch)

            number_coins_in_batch = len(coin_batch)

            if number_coins_in_batch == 0:

                break

            # get the detector of each coincidence pair

            dets1 = coin_batch[:, 2].astype(int)

            dets2 = coin_batch[:, 5].astype(int)

            thetas = coin_batch[:, -1]

            # get the x and y values of each coincidence pair.
            # these are relative to the detector origin!

            horizontal_coords_1 = coin_batch[:, 0].reshape(-1, 1)
            
            vertical_coords_1 = coin_batch[:, 1].reshape(-1, 1) # actually z in our coord space

            horizontal_coords_2 = coin_batch[:, 3].reshape(-1, 1) 

            vertical_coords_2 = coin_batch[:, 4].reshape(-1, 1) # actually z in our coord space

            

            # compute gamma1 coordinates

            O_vs1 = O_vs[dets1-1, :]
            #print(dets1)
            #print(O_vs1)
            #print(np.concatenate((dets1, O_vs1), axis = 1))

            H_vs1 = H_vs[dets1-1, :]

            V_vs1 = V_vs[dets1-1, :]
            
            #print(vertical_coords_1)
            #print(horizontal_coords_1-21.5)
            #print(O_vs1 + np.multiply(-(horizontal_coords_1-21.5), H_vs1) + np.multiply(vertical_coords_1-21.5, V_vs1))



            xyz1_coords = O_vs1 + np.multiply((horizontal_coords_1-21.5), H_vs1) + np.multiply(vertical_coords_1-21.5, V_vs1)
            #xyz1_coords = (xyz1_coords + np.multiply(xyz_shift, H_vs1) + np.multiply(xyz_shift, V_vs1))

            
            # compute gamma2 coordinates

            O_vs2 = O_vs[dets2-1, :]

            H_vs2 = H_vs[dets2-1, :]

            V_vs2 = V_vs[dets2-1, :]

            #FOR FUTURE EDITS: O_vs2 is the global position of the detector represented as a vector that points to the middle of the detector
            #H_vs2 are the unit vectors that point counterclockwise to the arial view. 

            xyz2_coords = O_vs2 + np.multiply((horizontal_coords_2-21.5), H_vs2) + np.multiply(vertical_coords_2-21.5, V_vs2)

            #xyz2_coords = (xyz2_coords + np.multiply(xyz_shift, H_vs2) + np.multiply(xyz_shift, V_vs2))

            # combine and rotate

            #zeros = np.zeros((len(xyz1_coords),1))
            
            coin_batch = np.concatenate((xyz1_coords, xyz2_coords), axis=1)
            #coin_batch = np.concatenate((xyz1_coords, zeros, zeros, xyz2_coords, zeros, zeros), axis=1)
            coin_batch = rotate_lines_about_z(coin_batch, thetas)

            #print(coin_batch)

            #print(coin_batch)

            #plotCoins(coin_batch)
            # combine coords and write

            coin_batch = coin_batch.astype(np.float32)

            coin_batch.tofile(output_fID)

            # update user if at checknpoint            

            coin_counter = coin_counter + number_coins_in_batch

            if coin_counter >= update_point:

                update_point = update_point + N*batch_size

                toc = time.perf_counter()

                t = (toc - tic)/60

                print(f"Coins processed: {coin_counter:,} in {t:,.2f} min")

            if coin_counter < batch_size:

                toc = time.perf_counter()

                t = (toc - tic)/60

                print(f"Coins processed: {coin_counter:,} in {t:,.2f} min")

                break
    
    print("Process complete.")
    return 0
   

def main():

    Directory = '07162025/'
    Project = ''
    matlab_raw_filename = 'Plant_raw_data_St_10_40am.E20..nCOT.RS(1).mat'
    coin_file = Directory + Project + matlab_raw_filename

    output_file = Directory + 'Out.txt'

    keys_of_interest = ['x1', 'y1', 'p1', 'x2', 'y2', 'p2', 'RA']

    #print_first_entries(coin_file, keys_of_interest)

    save_selected_keys_to_ascii(coin_file, output_file, keys_of_interest)

    #Converting the coins into ascii files
    coin_file = Directory + "Out.txt"

    det_file  = "/Users/wonupark/Desktop/PhytoPET/scripts/NewRingD162.d.txt"


    output_file = Directory + matlab_raw_filename + ".lm"
    batch_size = 1000
    update_user_every_N_batches = 1000
    xyz_shift = [-21.5, -21.5, -21.5]
    processCoins(coin_file, det_file, output_file, xyz_shift, batch_size, update_user_every_N_batches)


    return 0


if __name__ == "__main__":

    main()



"""

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

def process_chunk(chunk_data, O_vs, H_vs, V_vs):
    """
    Process a chunk of raw data entirely in memory.
    chunk_data expected columns: [x1, y1, det1, x2, y2, det2, theta]
    """
    
    # 1. Extract columns (using views to avoid copies where possible)
    # The hardcoded '21.5' is the center shift from your original logic
    h1 = chunk_data[:, 0][:, np.newaxis] - 21.5
    v1 = chunk_data[:, 1][:, np.newaxis] - 21.5
    d1 = chunk_data[:, 2].astype(int) - 1 # Adjust 1-based index to 0-based
    
    h2 = chunk_data[:, 3][:, np.newaxis] - 21.5
    v2 = chunk_data[:, 4][:, np.newaxis] - 21.5
    d2 = chunk_data[:, 5].astype(int) - 1
    
    thetas = chunk_data[:, 6]

    # 2. Vectorized Geometry Calculation
    # Fetch vectors for all events in this batch simultaneously
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
    # Structure: [x1, y1, z1, 0, 0, x2, y2, z2, 0, 0]
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
    output_file = Directory + matlab_raw_filename + ".lm"
    
    # Keys corresponding to: h1, v1, det1, h2, v2, det2, theta
    keys_of_interest = ['x1', 'y1', 'p1', 'x2', 'y2', 'p2', 'RA']
    
    BATCH_SIZE = 100000 # Increased from 1,000 to 100,000 for speed
    
    # --- Initialization ---
    try:
        os.remove(output_file)
    except OSError:
        pass

    print("Loading detector geometry...")
    O_vs, H_vs, V_vs = getDetVectors(det_filename)

    tic = time.perf_counter()

    # --- Main Pipeline ---
    with h5py.File(input_file, 'r') as f, open(output_file, 'wb') as out_f:
        
        # Access datasets (pointers only, not reading yet)
        datasets = [f[k] for k in keys_of_interest]
        total_entries = datasets[0].shape[1]
        
        print(f"Total events to process: {total_entries:,}")
        
        for start_idx in range(0, total_entries, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, total_entries)
            
            # 1. READ: Load a chunk of columns directly from HDF5
            # Stack them into shape (N, 7)
            # Note: We read [0, start:end] because your HDF5 shape is likely (1, N)
            chunk_data = np.stack([d[0, start_idx:end_idx] for d in datasets], axis=1).astype(np.float32)
            
            # 2. PROCESS: Compute geometry and rotations in memory
            processed_chunk = process_chunk(chunk_data, O_vs, H_vs, V_vs)
            
            # 3. WRITE: Append binary data to disk
            processed_chunk.tofile(out_f)
            
            # Feedback
            if (end_idx // BATCH_SIZE) % 5 == 0 or end_idx == total_entries:
                toc = time.perf_counter()
                elapsed_min = (toc - tic) / 60
                percent = (end_idx / total_entries) * 100
                print(f"Processed {end_idx:,}/{total_entries:,} ({percent:.1f}%) in {elapsed_min:.2f} min")

    print("Process complete.")

if __name__ == "__main__":
    main()