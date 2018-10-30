"""
===============
Projected Fermi surface
===============

"""

from fplore.loader import FPLORun, Band
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

run = FPLORun("../example_data/fermisurf")
axes, data = run['+band_kp'].reshaped_data

# axis to project along, 0: x, 1: y, 2: z
axis_to_project = 2

visible_axes = [0, 1, 2]
visible_axes.remove(axis_to_project)

axis_labels = [r'$k_x / \frac{2\pi}{a}$',
               r'$k_y / \frac{2\pi}{a}$',
               r'$k_z / \frac{2\pi}{c}$']

im = Band.smooth_overlap(data['e'], e=0, scale=0.04, axis=axis_to_project)
im = im.T  # so the first axis (x) is displayed horizontally not vertically

plt.imshow(im, extent=(axes[visible_axes[0]][0], axes[visible_axes[0]][-1],
                       axes[visible_axes[1]][0], axes[visible_axes[1]][-1]),
           interpolation="bicubic", origin='lower')
plt.xlabel(axis_labels[visible_axes[0]])
plt.ylabel(axis_labels[visible_axes[1]])
plt.tight_layout()
plt.show()

