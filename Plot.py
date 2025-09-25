import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.optimize import curve_fit
import os

# === Configuration ===
vox_filename = 'SummerPHYTOPET/CylinderTest_modified.vox'  # <-- Replace with your actual file
header_size = 20
shape = (300, 100, 300)  # (X, Y, Z)
dtype = np.float32
z1, z2 = 105, 195  # Z range for plotting fit

# === Gaussian with offset ===
def gaussian_with_offset(z, b, p, c, s):
    return b + p * np.exp(-0.5 * ((z - c) ** 2) / s ** 2)

# === Load .vox file ===
expected_size = np.prod(shape) * np.dtype(dtype).itemsize
file_size = os.path.getsize(vox_filename)
if file_size != expected_size + header_size:
    raise ValueError(f"File size mismatch: expected {expected_size + header_size}, got {file_size}")

with open(vox_filename, 'rb') as f:
    f.seek(header_size)
    data = np.frombuffer(f.read(), dtype=dtype)

volume = data.reshape(shape)

# === Plot setup ===
center_y = shape[1] // 2
x_min, x_max = 107, 193
x_init = (x_min + x_max) // 2
z_axis = np.arange(shape[2])

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.3)

profile = volume[x_init, center_y, :]
line, = ax.plot(z_axis, profile, lw=2, label="Z-profile")

vline1 = ax.axvline(x=z1, color='red', linestyle='--', label='Z = 80')
vline2 = ax.axvline(x=z2, color='green', linestyle='--', label='Z = 220')
gauss_line, = ax.plot([], [], color='orange', lw=2, label="Gaussian Fit")

param_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                     verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

ax.set_xlabel('Z')
ax.set_ylabel('Intensity')
ax.set_title(f'Z-profile at X={x_init}, Y={center_y}')
ax.grid(True)
ax.legend()

# === Slider ===
ax_slider = plt.axes([0.25, 0.15, 0.5, 0.03])
x_slider = Slider(ax_slider, 'X', x_min, x_max, valinit=x_init, valfmt='%0.0f')

def update(val):
    x = int(x_slider.val)
    profile = volume[x, center_y, :]
    line.set_ydata(profile)
    ax.set_title(f'Z-profile at X={x}, Y={center_y}')

    # Fit wider region than visible plot
    z_fit_start = max(z1 - 40, 0)
    z_fit_end = min(z2 + 40, shape[2])
    z_fit_full = np.arange(z_fit_start, z_fit_end)
    y_fit_full = profile[z_fit_start:z_fit_end]

    # Initial guess: baseline, peak, center, width
    b0 = np.min(y_fit_full)
    p0 = np.max(y_fit_full) - b0
    c0 = z_fit_full[np.argmax(y_fit_full)]
    s0 = 10
    try:
        popt, _ = curve_fit(gaussian_with_offset, z_fit_full, y_fit_full, p0=[b0, p0, c0, s0])
        b, p, c, s = popt

        z_plot = np.arange(z1, z2)
        fit_vals = gaussian_with_offset(z_plot, *popt)
        gauss_line.set_data(z_plot, fit_vals)

        param_text.set_text(f"Fit: b={b:.2f}, p={p:.2f}, μ={c:.2f}, σ={s:.2f}")
    except RuntimeError:
        gauss_line.set_data([], [])
        param_text.set_text("Fit failed")

    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw_idle()

x_slider.on_changed(update)

plt.show()