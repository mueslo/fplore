"""
===============
Band projection
===============

"""

import numpy as np
import matplotlib.pyplot as plt

from fplore.loader import FPLORun
from fplore.project import project

run = FPLORun("../example_data/fermisurf")

X_point = np.array((-0.5, 0, 0))
M_point = np.array((-0.5, -0.5, 0))
point_3 = np.array((0, 0, -0.5))

level_indices = run["+band_kp"].bands_within(-0.25, 0.25)
print("Plotting", len(level_indices), " projected bands.")
ip = run["+band_kp"].interpolator

path = run.linspace_ng(X_point, M_point, point_3,
                       num=(250, 50))
bands_along_path = ip(path)

show_bz = False
n = 2 + show_bz

f = plt.figure(figsize=(n*4, 4))
ax1 = f.add_subplot(1, n, 1)
ax2 = f.add_subplot(1, n, 2, sharex=ax1, sharey=ax1)

if show_bz:
    ax3 = f.add_subplot(1, n, 3, projection='3d')

    points = path.reshape(-1, 3)
    run.plot_bz(ax3, k_points=True, high_symm_points=False)
    ax3.plot(*run.frac_to_k(points).T, color='k', alpha=0.3, label="projected k-points")

    ax3.set_title('projected part of BZ')
    ax3.legend(loc='lower right')

for idx_e in level_indices:
    bap = bands_along_path[:, :, idx_e]
    i = np.linspace(0, 1, bap.shape[0])
    j = np.linspace(0, 1, bap.shape[1])
    ij = np.meshgrid(i, j, indexing='ij')
    ax1.plot(ij[0], bap, color='gray', lw=0.1)
ax1.set_title('naive line plot')

for idx_e in level_indices:  # range(e_k_3d.shape[3]):
    bap = bands_along_path[:, :, idx_e]
    i = np.linspace(0, 1, bap.shape[0])
    j = np.linspace(0, 1, bap.shape[1])
    i, j = np.meshgrid(i, j, indexing='ij')
    pc = project(i, j, bap, axis=1, color=(0.5, 0.5, 0.5, 1.0))
    ax2.add_collection(pc)
ax2.set_title('using fplore.project')

plt.show()
