"""
==============
Brillouin zone
==============

"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc

from fplore.loader import FPLORun

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

run = FPLORun("../example_data/fermi")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
run.plot_bz(ax, k_points=True, use_symmetry=True)

plt.show()
