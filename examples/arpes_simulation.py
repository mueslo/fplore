"""
================
ARPES simulation
================

"""

import numpy as np
from fplore import FPLORun
from fplore.util import k_arpes, rot_v1_v2, sample_e
from fplore.plot import plot_bz_proj
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

run = FPLORun("../example_data/Ag")


surface_normal = [1, 1, 1]  # todo conventional to primitive
theta = np.linspace(-np.radians(20.8), np.radians(20.8), 200)
theta2 = np.array([np.radians(3.49-3.49)])
v0 = 12
workfunc_analyzer = 4
e_photon = 665

R = rot_v1_v2(surface_normal, [0, 0, 1])
phi = np.radians(-15)
c, s = np.cos(phi), np.sin(phi)
R = np.dot(np.array(((c, -s, 0), (s, c, 0), (0, 0, 1))), R)
ex, ey, ez = R

# todo: order of rotation operations
# todo: setup_arpes_geometry(photon momentum, angles, ...)

ax1 = plt.subplot(121)
plot_bz_proj(run, ax1, rot=R, axis=1, color='grey', lw=0.1,
             neighbours=[[5, 0, 0], [4, 0, 0], [3, 0, 0], [2, 0, 0],[1, 0, 0]])


k_par, k_par2, k_perp = k_arpes(theta, e_photon-workfunc_analyzer, v0, theta2)
ax1.plot(k_par, k_perp, label='sampled k-points')

print(k_par2)

# since y was projected:
k = (ex * k_par[:, np.newaxis] +
     ey * k_par2[:, np.newaxis] +
     ez * k_perp[:, np.newaxis])

print("kpar=", ex)
print("kperp=", ez)
print("in/plane=", ey)

k_1bz = run.backfold_k(k)
P = R[[True, False, True]]
k_1bz_2d = np.dot(P, k_1bz.T)

dists = np.linalg.norm(np.diff(k_1bz, axis=0), axis=1)
dists_ = np.linalg.norm(np.diff(k, axis=0), axis=1)
# insert nans to make BZ folds discontinuous
k_1bz_2d = np.insert(k_1bz_2d, np.where(dists > 1.01*dists_)[0]+1,
                     np.array([[np.nan, np.nan]]).T, axis=1)
ax1.plot(*k_1bz_2d, label='backfolded sampled\nk-points')

ax1.set_xlabel(r'$k_\| / \mathrm{\AA}^{-1}$',)
ax1.set_ylabel(r'$k_\perp / \mathrm{\AA}^{-1}$')
ax1.legend(loc='center')

ax2 = plt.subplot(122)

axes, idx_grid = run.band.reshape_gridded_data(missing_coords_strategy='backfold')
energy_window = (-10., 1.)
level_indices = run.band.bands_within(*energy_window)
bands_to_sample = run.band.data['e'][..., level_indices]

grid_data = bands_to_sample[idx_grid]
path = k_1bz
bands_along_path = sample_e(axes, grid_data, path, order=1)
ax2.plot(theta*180/np.pi, bands_along_path, color='k', lw=0.5)
ax2.set_ylabel(r'Binding energy $E-E_\mathrm{F}/\mathrm{eV}$')
ax2.set_xlabel(r'Polar angle $\theta/\mathrm{deg}$')
ax2.set_ylim(*energy_window)

plt.suptitle(r'Ag(111), $h\nu = {}\,\mathrm{{eV}}$'.format(e_photon))
plt.show()

