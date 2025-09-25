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
    


def plotCoins(coin_pairs):


    x1 = coin_pairs[:, 0]

    y1 = coin_pairs[:, 1]

    z1 = coin_pairs[:, 2]

    x2 = coin_pairs[:, 5]

    y2 = coin_pairs[:, 6]

    z2 = coin_pairs[:, 7]
    


    ax = plt.axes(projection='3d') 

    for i in range(0,len(x1),4):
        ax.plot([x1[i], x2[i]], [y1[i], y2[i]], [z1[i], z2[i]], color='blue', linewidth = 0.3)


    #ax.scatter(x1, y1, z1, color='r', alpha=0.8, s=3.0)

    #ax.scatter(x2, y2, z2, color='g', alpha=0.8, s=3.0)

    ax.set_xlabel('X')

    ax.set_ylabel('Y')

    ax.set_zlabel('Z')

    ax.set_box_aspect(aspect=[2, 2, 0.5] , zoom=1.2)

    plt.show()

    return 0

    
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

            print(coin_batch)

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

    Directory = 'SummerPHYTOPET/'
    Project = ''
    matlab_raw_filename = 'Uniform_Cylinder_30mins_250UCi_4_14pm_rot30secc.E20..nCOT.RS.mat'
    coin_file = Directory + Project + matlab_raw_filename
    #Annulus_Only_260uCi_ST_1133am_rot4.E20..nCOT.RS.mat
    #04182025_3PS_Fucdicials.E15..nCOT.RS.mat
    #plant_glucose_PET_data_06262025.E20..nCOT.RS.mat

    #Uniform_Cylinder_30mins_250UCi_4_14pm_rot30secc.E20..nCOT.RS.mat
    #PS_2min_test_start_rot_30sec_1.E20..TB.nCOT.RS.mat
    output_file = Directory + 'Out.txt'

    keys_of_interest = ['x1', 'y1', 'p1', 'x2', 'y2', 'p2', 'RA']

    #print_first_entries(coin_file, keys_of_interest)

    save_selected_keys_to_ascii(coin_file, output_file, keys_of_interest)

    #Converting the coins into ascii files
    coin_file = Directory + "Out.txt"

    det_file  = Directory + 'NewRingD162.d.txt'

    output_file = Directory + Project + "Cylinder_norm" + ".lm"
    batch_size = 1000
    update_user_every_N_batches = 1000
    xyz_shift = [-21.5, -21.5, -21.5]
    processCoins(coin_file, det_file, output_file, xyz_shift, batch_size, update_user_every_N_batches)


    return 0


if __name__ == "__main__":

    main()



