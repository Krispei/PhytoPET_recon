import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import tifffile

def plot_3d_tiff(filepath, colormap='viridis', alpha_threshold=0.1):
    """
    Plot all z-stacks of a 3D TIFF file with transparency for low values.
    
    Parameters:
    -----------
    filepath : str
        Path to the 3D TIFF file
    colormap : str
        Matplotlib colormap name (default: 'viridis')
    alpha_threshold : float
        Values below this (normalized 0-1) become transparent
    """
    
    # Load the 3D TIFF file
    print(f"Loading TIFF file: {filepath}")
    img_stack = tifffile.imread(filepath)
    
    print(f"Image shape: {img_stack.shape}")
    print(f"Data type: {img_stack.dtype}")
    print(f"Value range: {img_stack.min()} to {img_stack.max()}")
    
    # Get dimensions
    if len(img_stack.shape) == 3:
        nz, ny, nx = img_stack.shape
    else:
        raise ValueError("Expected 3D TIFF file")
    
    # Normalize data for colormap
    '''
    vmin, vmax = img_stack.min(), img_stack.max()
    if vmax > vmin:
        img_normalized = (img_stack - vmin) / (vmax - vmin)
    else:
        img_normalized = img_stack
    '''
        

    # Create figure with 3D subplot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    cmap = cm.get_cmap(colormap)
    
    print("Plotting all z-stacks...")
    
    # Plot every z-stack
    for z in range(nz):
        if z % 10 == 0:  # Progress indicator
            print(f"Processing z-slice {z}/{nz}")
        
        slice_data = img_stack[z]
        
        # Create coordinate grids
        y, x = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        
        # Filter out near-zero values
        mask = slice_data > alpha_threshold
        
        if not np.any(mask):
            continue
        
        x_plot = x[mask]
        y_plot = y[mask]
        z_plot = np.full(x_plot.shape, z)
        colors = slice_data[mask]
        
        # Calculate alpha values: opaque above threshold, transparent below
        alphas = np.ones_like(colors)
        
        # Convert colors to RGBA
        rgba = cmap(colors)
        rgba[:, 3] = alphas  # Set alpha channel
        
        # Plot scatter points
        ax.scatter(x_plot, y_plot, z_plot, c=rgba, s=1, edgecolors='none')
    

    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D TIFF Volume Visualization\nAll {nz} z-stacks')
    
    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap)
    cbar = plt.colorbar(sm, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Intensity', rotation=270, labelpad=15)
    
    plt.tight_layout()
    plt.show()
    
    print("Visualization complete!")


if __name__ == "__main__":
    # Example usage
    filepath = "img/09182025_pproj_3min_40itr_mov/09182025_66min.lm_041.tiff"  # Replace with your TIFF file path
    
    # Plot with default settings
    # alpha_threshold controls transparency: lower = more voxels visible
    plot_3d_tiff(filepath, colormap='viridis', alpha_threshold=0.1)
    
    # Try different colormaps: 'viridis', 'plasma', 'inferno', 'magma', 'jet', 'hot', 'cool'
    # Adjust alpha_threshold (0.0 to 1.0) to control which values are visible