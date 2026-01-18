import numpy as np
import h5py
import os
import shutil
import matplotlib.pyplot as plt
from itertools import islice
from pathlib import Path

def getDetVectors(det_file):
    det_vectors = np.loadtxt(det_file, dtype=float, delimiter=',')

    # Ensure the detector file was loaded correctly as 2D (N x 3)
    if det_vectors.ndim == 1:
        # Expecting multiples of 3 columns (x, y, z per vector)
        if det_vectors.size % 3 != 0:
            raise ValueError(f"Detector file seems malformed: total size {det_vectors.size} not divisible by 3")
        det_vectors = det_vectors.reshape((-1, 3))

    # O_vs gets the "theta vector of each detector"
    O_vs = det_vectors[0::3, :]
    # H_vs gets the "azimuthal vector of each detector"
    H_vs = det_vectors[1::3, :]
    # V_vs is the vertical vector (e.g. [0,0,1])
    V_vs = det_vectors[2::3, :]

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

def processCoins(output_file, coin_batch, O_vs, H_vs, V_vs):

    # remove previous output files by the same name

    try:

        os.remove(output_file)

    except OSError:

        pass

    # initialize

    with open(output_file, 'ab') as output_fID:

        # get the detector of each coincidence pair

        dets1 = coin_batch[:, 2].astype(int)

        dets2 = coin_batch[:, 5].astype(int)

        thetas = coin_batch[:, -2]

        #num_dets = O_vs.shape[0]  # usually 162 for NewRingD162

        horizontal_coords_1 = coin_batch[:, 0].reshape(-1, 1)
        
        vertical_coords_1 = coin_batch[:, 1].reshape(-1, 1) # actually z in our coord space

        horizontal_coords_2 = coin_batch[:, 3].reshape(-1, 1) 

        vertical_coords_2 = coin_batch[:, 4].reshape(-1, 1) # actually z in our coord space

        # compute gamma1 coordinates

        O_vs1 = O_vs[dets1-1, :]
     
        H_vs1 = H_vs[dets1-1, :]

        V_vs1 = V_vs[dets1-1, :]

        xyz1_coords = O_vs1 + np.multiply((horizontal_coords_1-21.5), H_vs1) + np.multiply(vertical_coords_1-21.5, V_vs1)
        
        # compute gamma2 coordinates

        O_vs2 = O_vs[dets2-1, :]

        H_vs2 = H_vs[dets2-1, :]

        V_vs2 = V_vs[dets2-1, :]

        xyz2_coords = O_vs2 + np.multiply((horizontal_coords_2-21.5), H_vs2) + np.multiply(vertical_coords_2-21.5, V_vs2)

        # combine and rotate
        
        coin_batch = np.concatenate((xyz1_coords, xyz2_coords), axis=1)

        coin_batch = rotate_lines_about_z(coin_batch, thetas)

        coin_batch = coin_batch.astype('<f4')
        
        coin_batch.tofile(output_fID)
        
    
    print("Process complete.")
    return 0

def save_selected_keys_to_ascii_chunked(ID, file_path, keys_of_interest, time, geometry_path):

    #Chnage this to where the detector file and input data is
    
    #listmode Output goes into this folder

    out_folder_path = str(Path(file_path).parent) + "/" + f"{ID}_dyn_{time}min_lm/"

    #Get the full paths

    # -------Load detector file-------

    O_vs, H_vs, V_vs = getDetVectors(geometry_path)


    with h5py.File(file_path, 'r') as f:

        num_entries = f[keys_of_interest[0]].shape[1]

        num_keys = len(keys_of_interest)

        # Preload all key data into memory

        all_key_data = {key: f[key][0, :] for key in keys_of_interest}

        time_stamp = time * 60000
        start_indx = 0
        end_indx = 0
        i = all_key_data["t"][end_indx]
        time_segments_processed = i// time_stamp
        chunks_processed = 0
        

        # -------Create Folder----------

        #tries deleting the folder
        if os.path.exists(out_folder_path):
            try:
                shutil.rmtree(out_folder_path)
                print(f"Folder '{out_folder_path}' and its contents deleted successfully.")
            except OSError as e:
                print(f"Error: {e}")
        else:
            print(f"Folder '{out_folder_path}' does not exist.")

        # Create the directory
        try:
            os.mkdir(out_folder_path)
            print(f"Directory '{out_folder_path}' created successfully.")
        except FileExistsError:
            print(f"Directory '{out_folder_path}' already exists.")
        except PermissionError:
            print(f"Permission denied: Unable to create '{out_folder_path}'.")
        except Exception as e:
            print(f"An error occurred: {e}")

        # --------------------------------

        while True:

            if ( i // time_stamp > time_segments_processed or end_indx == num_entries-1):
                print(f"{i} {i//time_stamp} {time_segments_processed} {start_indx} {end_indx}")
                #print(end_indx)
                time_segments_processed += 1
                i = i % time_stamp

                chunk_raw = {key: values[start_indx: end_indx] for key, values in all_key_data.items()}

                chunk_array = np.zeros(shape=((end_indx-start_indx), num_keys))
                
                for key_idx, key in enumerate(chunk_raw):

                    chunk_array[:, key_idx] = chunk_raw[key][:].flatten()


                output_file = out_folder_path + f"{chunks_processed*(time_stamp//60000)}min.lm"
                
                processCoins(output_file, chunk_array, O_vs, H_vs, V_vs)

                #increment the indicies
                chunks_processed+=1
                print(f'chunks of {time_stamp} ms processed: {chunks_processed+1}')

                start_indx = end_indx
                end_indx = start_indx + 1
            else:
                end_indx = end_indx + 1

            if (end_indx >= num_entries):
                break
            
            i = all_key_data['t'][end_indx]

def main():

    ID = '06242025'
    RAW_DATA_PATH = r"C:\Users\Krisps\PHYTOPet\06242025\Plant_06242025.E20..nCOT.RS.mat"
    DET_GEOMETRY_PATH = r"C:\Users\Krisps\PHYTOPet\PhytoPET_recon\NewRingD162.d.txt"
    TIME_BINS = 3


    keys_of_interest = ['x1', 'y1', 'p1', 'x2', 'y2', 'p2', 'RA', 't']

    save_selected_keys_to_ascii_chunked(ID, RAW_DATA_PATH, keys_of_interest, TIME_BINS, DET_GEOMETRY_PATH)

    #Remove files under 20kb

if __name__ == "__main__":

    main()
        


