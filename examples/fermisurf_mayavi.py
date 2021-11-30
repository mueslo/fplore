"""
===============
Fermi surface
===============

"""

from mayavi import mlab
from fplore import FPLORun

run = FPLORun("../example_data/yrs")
level_indices = run.band.bands_at_energy()
energy_data = run.band.data['e'][..., level_indices]

axes, grid_idx = run.band.reshape_gridded_data()
data = energy_data[grid_idx]  # reshape to grid

mlab.figure()

for i, level_idx in enumerate(level_indices):
    mlab.contour3d(data[..., i], contours=[0], opacity=0.4)

mlab.show()
