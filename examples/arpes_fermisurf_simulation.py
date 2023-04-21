"""
=================
ARPES Fermi surface map simulation
=================

"""

import numpy as np
from scipy.stats import norm

from fplore import FPLORun
from fplore.util import k_arpes, rot_v1_v2
from fplore.plot import plot_bz
from fplore.arpes import ARPESGeometry
import matplotlib.pyplot as plt

kz_broadening = True
kz_curvature = True
e_broadening = True
fermi_cutoff = False
kxky_broadening = True


run = FPLORun("../example_data/Ag")
surface_normal = run.high_symm_kpoints['L']  # 111
lens_axis = run.high_symm_kpoints['X']  # 100
# vector rejection of normal to get perpendicular component to surface normal:
lens_axis = lens_axis - (lens_axis @ surface_normal) / (surface_normal @ surface_normal) * surface_normal

thetacenter, thetawidth = 0, 32
mintheta, maxtheta = thetacenter - thetawidth / 2, thetacenter + thetawidth / 2
theta = np.linspace(np.deg2rad(mintheta), np.deg2rad(maxtheta), 400)

theta2center, theta2width = 0, 24  # 1.3, 12
mintheta2, maxtheta2 = theta2center - theta2width / 2, theta2center + theta2width / 2
theta2 = np.linspace(np.deg2rad(mintheta2), np.deg2rad(maxtheta2), 200)

extent = (mintheta, maxtheta, mintheta2, maxtheta2)
theta, theta2 = np.meshgrid(theta, theta2)

v0 = 8
workfunc_analyzer = 3.8
e_photon = 670

if kz_broadening:
    imfp = 10  # angstroms
    Delta_kz = 1 / imfp  # standard deviation in reciprocal angstroms (∆o)² = <o²> - <o>²
    n_kz = 47  # 13
    kz_deltas = np.linspace(-3 * Delta_kz, 3 * Delta_kz, n_kz)  # shape (N,)
    kz_weights_pdf = norm.cdf(kz_deltas, scale=Delta_kz)
    kz_limits = np.hstack([[-np.inf], (kz_deltas[1:] + kz_deltas[:-1]) / 2, [np.inf]])
    kz_weights = np.array([norm.cdf(u, scale=Delta_kz) - norm.cdf(l, scale=Delta_kz)
                           for l, u in zip(kz_limits, kz_limits[1:])])
    weights = kz_weights.reshape((1, 1, n_kz, 1))

    kz_deltas = np.pad(kz_deltas[np.newaxis].T, [(0, 0), (2, 0)])[np.newaxis, np.newaxis]  # to shape (1, 1, N, 3)
else:
    kz_deltas = 0
    weights = 1

R = rot_v1_v2(surface_normal, [0, 0, 1])
R = rot_v1_v2(R @ lens_axis, [1, 0, 0]) @ R  # maps cartesian coords of run.band to cartesian coords of kxkykz
R = R.T  # for multiplying from the right (calculation_coords @ R -> msmt_aligned_coords)

ag = ARPESGeometry(angles_photon=(-60 * np.pi / 180, 0),
                   angles_sample=(0, 0),
                   slit_direction=(0, 0, 1))

k_p_mean = k_arpes(theta=theta, e_photon=e_photon, phi_det=workfunc_analyzer, v0=v0, theta2=theta2, geometry=ag)
if not kz_curvature:
    k_p_mean[:, :, 2] = k_p_mean[:, :, 2].max()
k_p = k_p_mean[:, :, np.newaxis] + kz_deltas  # add kz broadening sample points in 3rd dimension
k = k_p @ R.T  # aligned -> calculation
k_1bz = run.backfold_k(k)


fig = plt.figure(figsize=(5, 6), constrained_layout=True, dpi=196)
#fig.subplots_adjust(wspace=0.6, hspace=0.6)
ax_bz = fig.add_subplot(2, 1, 1, projection='3d')
ax_bz.plot_surface(*(k_p_mean[::40, ::40]).T,
                   alpha=0.3)  # don't draw all the points  # - [0, 0, 8] @ run.primitive_lattice.reciprocal_lattice.matrix @ R
midpoint = k.mean(axis=(0, 1, 2))
offset = midpoint @ run.primitive_lattice.reciprocal_lattice.inv_matrix
offset = np.rint(offset)
plot_bz(run, ax_bz, k_points=False, vectors=None, offset=offset, rot=R)
ax_bz.view_init(elev=15., azim=-81.)

ax = fig.add_subplot(2, 1, 2)
ax.set_xlabel(r'$\theta_\mathrm{Lens} / \mathrm{deg}$', )
ax.set_ylabel(r'$\theta_\mathrm{Deflection} / \mathrm{deg}$')


level_indices = run.band.bands_within(0.5)  # (*energy_window)

print('Getting interpolator...')
ip = run.band.get_interpolator(bands=level_indices)
print('Got interpolator. Doing interpolation...')
data = ip(k_1bz)

if not e_broadening:
    scale = 0.01  # FWHM, eV
else:
    scale = 0.1  # FWHM, eV
scale = scale / 2.355  # FWHM -> std

if fermi_cutoff:
    # can generate kz artefacts, since every time a isosurface at different kz crosses ef
    # intensity will suddenly drop
    T = 40  # K
    k_B = 8.617 * 10 ** -5  # eV/K
    weights = weights * (1 / (np.exp(data / (k_B * T)) + 1))

im = run.band.smooth_overlap(data, e=0, scale=scale, axis=2, weights=weights)

if kxky_broadening:
    from scipy.ndimage import gaussian_filter
    im = gaussian_filter(im, (1, 1))

ax.imshow(im.T, extent=extent, aspect='equal')  # .T so the first axis (x) is displayed horizontally not vertically
plt.suptitle(f'Ag ARPES Fermimap (001), $h\\nu = {e_photon}\\,\\mathrm{{eV}}$\nwith $k_z$ curvature and broadening')

ax_bz.set_box_aspect([ub - lb for lb, ub in (getattr(ax_bz, f'get_{a}lim')() for a in 'xyz')])
plt.show()
