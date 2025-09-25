import numpy as np
import napari


# Adjust these values:
#/Users/wonupark/Desktop/PhytoPETwork/SummerPHYTOPET/AnnulusTest_norm_no_tof.vox

filename = "SummerPHYTOPET/AnnulusTest_norm_no_tof.vox"
dtype = np.float32
shape = (300,100,300)

data = np.fromfile(filename, dtype=dtype)
volume = data[5:].reshape(shape)
# Now you can view or modify the volume
volume[volume < 0] = 0

print(volume[150,50,:])

viewer = napari.Viewer(ndisplay=3)
viewer.add_image(volume, colormap='gray')
napari.run()



# Save back to .vox
#volume.astype(dtype).tofile("FIXED_CylinderTest_norm_no_tof.vox")