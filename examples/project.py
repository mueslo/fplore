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

axes, data = run['+band_kp'].reshaped_data
level_indices = run["+band_kp"].bands_at_energy()

ip = run["+band_kp"].interpolator

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, sharex=True)

path = run.linspace_ng(X_point, M_point, np.array((0, 0, -0.5)),
                          num=(250, 50))

bands_along_path = ip(path)

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
    pc = project(i, j, bap, axis=1)
    ax2.add_collection(pc)
ax2.set_title('using fplore.project')

plt.show()
