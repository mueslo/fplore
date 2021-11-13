"""
==============
Brillouin zone
==============

"""

import matplotlib.pyplot as plt

from fplore import FPLORun
from fplore.plot import plot_bz

run = FPLORun("../example_data/fermi")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_bz(run, ax, k_points=True, use_symmetry=True)

plt.show()
