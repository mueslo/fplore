"""
===============
Fermi surface
===============

"""

from mayavi import mlab
from fplore import FPLORun

run = FPLORun("../example_data/fermisurf")
axes, data = run.band.reshape_gridded_data()
level_indices = run.band.bands_at_energy()

mlab.figure()

for level_idx in level_indices:
    mlab.contour3d(data['e'][:, :, :, level_idx], contours=[0], opacity=0.4)

mlab.show()
