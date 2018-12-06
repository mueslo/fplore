"""
===============
Fermi surface
===============

"""

from mayavi import mlab
from fplore import FPLORun

run = FPLORun("../example_data/fermisurf")
level_indices = run.band.bands_at_energy()
axes, data = run.band.reshape_gridded_data(energy_levels=level_indices)

mlab.figure()

for i, level_idx in enumerate(level_indices):
    mlab.contour3d(data['e'][..., i], contours=[0], opacity=0.4)

mlab.show()
