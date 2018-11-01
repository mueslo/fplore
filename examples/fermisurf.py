"""
===============
Fermi surface
===============

"""

from mayavi import mlab
from fplore.loader import FPLORun

run = FPLORun("../example_data/fermisurf")
axes, data = run['+band_kp'].reshaped_data
level_indices = run["+band_kp"].bands_at_energy()

mlab.figure()

for level_idx in level_indices:
    mlab.contour3d(data['e'][:, :, :, level_idx], contours=[0], opacity=0.4)

mlab.show()
