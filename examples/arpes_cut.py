"""
=================
ARPES k-space cut
=================

"""

import numpy as np
from fplore import FPLORun
from fplore.util import k_arpes, rot_v1_v2, sample_e
from fplore.plot import plot_bz_proj
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.gridspec as gridspec

run = FPLORun("../example_data/Ag")

surface_normal = [1, 1, 1]  # todo conventional to primitive

thetacenter = 0
thetawidth = 20
theta = np.linspace(np.radians(thetacenter-thetawidth/2), np.radians(thetacenter+thetawidth/2), 200)
theta2 = np.array([np.radians(0)])
v0 = 8
workfunc_analyzer = 3.8
e_photon = 674.4

R = rot_v1_v2(surface_normal, [0, 0, 1])
phi = np.radians(15)
c, s = np.cos(phi), np.sin(phi)
R = np.dot(np.array(((c, -s, 0), (s, c, 0), (0, 0, 1))), R)
ex, ey, ez = R

# todo: order of rotation operations
# todo: setup_arpes_geometry(photon momentum, angles, ...)

plt.figure(figsize=(8, 8))
gs = gridspec.GridSpec(3, 3)
ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[1:, 0])
ax3 = plt.subplot(gs[:, 1:])

plot_bz_proj(run, ax1, rot=R, axis=2, color='grey', lw=0.1)
ax1.set_xlabel(r'$k_\| / \mathrm{\AA}^{-1}$',)
ax1.set_ylabel(r'$k_{\|2} / \mathrm{\AA}^{-1}$')


plot_bz_proj(run, ax2, rot=R, axis=1, color='grey', lw=0.1, neighbours=[])


k_par, k_par2, k_perp = k_arpes(theta, e_photon-workfunc_analyzer, v0, theta2)
ax2.plot(k_par, k_perp, label='sampled k-points')

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

P1 = R[[True, True, False]]
k_1bz_2d_1 = np.dot(P1, k_1bz.T)

dists = np.linalg.norm(np.diff(k_1bz, axis=0), axis=1)
dists_ = np.linalg.norm(np.diff(k, axis=0), axis=1)
# insert nans to make BZ folds discontinuous
k_1bz_2d = np.insert(k_1bz_2d, np.where(dists > 1.01*dists_)[0]+1,
                     np.array([[np.nan, np.nan]]).T, axis=1)
ax1.plot(*k_1bz_2d_1)
ax2.plot(*k_1bz_2d, label='backfolded sampled\nk-points')

ax2.set_xlabel(r'$k_\| / \mathrm{\AA}^{-1}$',)
ax2.set_ylabel(r'$k_\perp / \mathrm{\AA}^{-1}$')
ax2.legend(loc='center')

axes, idx_grid = run.band.reshape_gridded_data(
    missing_coords_strategy='backfold')
energy_window = (-10., 1.)
level_indices = run.band.bands_within(*energy_window)
bands_to_sample = run.band.data['e'][..., level_indices]

grid_data = bands_to_sample[idx_grid]
path = k_1bz
bands_along_path = sample_e(axes, grid_data, path, order=1)
ax3.plot(theta*180/np.pi, bands_along_path, color='k', lw=0.5)
ax3.set_ylabel(r'Binding energy $E-E_\mathrm{F}/\mathrm{eV}$')
ax3.set_xlabel(r'Polar angle $\theta/\mathrm{deg}$')
ax3.set_ylim(*energy_window)

plt.suptitle(r'Ag(111), $h\nu = {}\,\mathrm{{eV}}$'.format(e_photon))
plt.show()
