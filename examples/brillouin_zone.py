"""
==============
Brillouin zone
==============

"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc

from fplore import FPLORun
from fplore.plot import plot_bz

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

run = FPLORun("../example_data/fermi")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plot_bz(run, ax, k_points=True, use_symmetry=True)

plt.show()
