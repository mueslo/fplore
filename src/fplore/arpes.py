from typing import NamedTuple, Optional
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation as R
from scipy.constants import c, hbar, eV, m_e, angstrom
import matplotlib.pyplot as plt

from .util import project_plane, normalized
from .logging import log
from .plot import Arrow3D

x, y, z = np.eye(3)
SphereCoordinate = NamedTuple('SphereCoordinate', [('phi', float), ('theta', float)])

# 3 levels of coordinate system:
# 1) geometry/lab frame coordinates ("global")
# 2) analyzer frame coordinates (x || slit, z || analyzer direction)
# 3.1) sample (surface) coordinates (z perp surface, x projection of slit onto surface)
# 3.2) crystal coordinates (lattice orientation)

# todo: refractive index of sample, usually not an issue at high energies since there n ~= 1.
#       see also: http://gisaxs.com/index.php/Materials. Might be relevant for Laser/(V)UV ARPES.

class ARPESGeometry(object):
    photon_direction: npt.ArrayLike = np.empty(3)
    slit_direction: npt.ArrayLike = np.empty(3)
    analyzer_direction: npt.ArrayLike = np.empty(3)

    def __init__(self, photon_direction: npt.ArrayLike,
                       slit_direction: npt.ArrayLike,
                       analyzer_direction: npt.ArrayLike):
        """
        Defines an ARPES endstation geometry in the laboratory frame.

        Conventionally z is up, but this is not required.

        :param photon_direction: (x, y, z) [vector] of the photon direction.
            Note that this is the Poynting vector.

        :param slit_direction: (x, y, z) [vector] of the slit direction of the
            analyzer in the lab frame (direction of positive angles).

        :param analyzer_direction: (x, y, z) [vector] of the analyzer direction in the lab frame.

        """
        assert np.isclose(np.dot(analyzer_direction, slit_direction), 0)

        self.photon_direction = normalized(np.array(photon_direction))
        self.slit_direction = normalized(np.array(slit_direction))
        self.analyzer_direction = normalized(np.array(analyzer_direction))

    @classmethod
    def from_angles(cls, photon_direction: SphereCoordinate,  # (phi, theta)
                    slit_direction: SphereCoordinate,
                    analyzer_direction: SphereCoordinate,
                    degrees: bool = True):
        invtheta = lambda sc: (sc[0], -sc[1])  # so that positive theta -> positive z
        return cls(
            photon_direction=R.from_euler('ZY', invtheta(photon_direction), degrees=degrees).apply(x),
            slit_direction=R.from_euler('ZY', invtheta(slit_direction), degrees=degrees).apply(x),
            analyzer_direction=R.from_euler('ZY', invtheta(analyzer_direction), degrees=degrees).apply(x),
        )

    @property
    def inv_analyzer_frame(self):
        """
        Returns the analyzer frame (in lab frame coordinates).

        analyzer_coord(3,) @ inv_analyzer_frame(3,3) -> lab_coord(3,)
        inv_analyzer_frame(3,3) @ lab_coord(3,) -> analyzer_coord(3,)
        """
        return np.array([
            self.slit_direction,
            np.cross(self.slit_direction, self.analyzer_direction),
            self.analyzer_direction])

    @property
    def analyzer_frame(self):
        """
        Returns the inverse analyzer frame (in lab frame coordinates).

        lab_coord(3,) @ analyzer_frame(3,3) -> analyzer_coord(3,)
        analyzer_frame(3,3) @ analyzer_coord(3,) -> lab_coord(3,)
        """
        return self.inv_analyzer_frame.T

    def get_inv_sample_frame(self, sample_normal_direction: Optional[npt.ArrayLike] = None):
        """
        Returns the inverse sample frame (in lab frame coordinates).

        sample_coord(3,) @ inv_sample_frame(3,3) -> lab_coord(3,)
        inv_sample_frame(3,3) @ lab_coord(3,) -> sample_coord(3,)
        """
        if sample_normal_direction is None:
            sample_normal_direction = self.analyzer_direction
        sample_normal_direction = normalized(sample_normal_direction)
        lensdir = normalized(
            project_plane(self.slit_direction,
                          sample_normal_direction))
        lensdir2 = np.cross(lensdir,
                            sample_normal_direction)
        R = np.array([lensdir,
                         lensdir2,
                         sample_normal_direction])
        assert np.allclose(R.T @ R, np.eye(3))
        return R

    def get_sample_frame(self, sample_normal_direction: Optional[npt.ArrayLike] = None):
        """Returns the sample frame (in lab frame coordinates).

        lab_coord(3,) @ sample_frame(3,3) -> sample_coord(3,)
        sample_frame(3,3) @ sample_coord(3,) -> lab_coord(3,)
        """
        return self.get_inv_sample_frame(sample_normal_direction).T

    def get_sample_frame_analyzer_coords(self, sample_normal_direction: Optional[npt.ArrayLike] = None):
        """Returns the sample frame (in analyzer frame coordinates).

        analyzer_coord(3,) @ sample_frame_analyzer_coords(3,3) -> sample_coord(3,)
        sample_frame_analyzer_coords(3,3) @ sample_coord(3,) -> analyzer_coord(3,)
        """
        return self.inv_analyzer_frame @ self.get_sample_frame(sample_normal_direction)

    def photon_momentum(self, e_photon: float):
        """Returns the photon momentum in the lab frame in reciprocal angstroms."""
        return self.photon_direction * e_photon * angstrom * eV / (hbar * c)

    def photon_momentum_sample_coords(self, e_photon: float, sample_normal_direction: Optional[npt.ArrayLike] = None):
        """Returns the photon momentum in sample frame coordinates.
        Defaults to perfect normal emission."""

        # multiplying from the right is like transverse/inverse multiplication from the left
        return self.photon_momentum(e_photon) @ self.get_sample_frame(sample_normal_direction)

    def plot(self, ax: Optional[plt.Axes] = None):
        if ax is None:
            fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

        ax.add_artist(Arrow3D(*np.vstack([-self.photon_direction, [0, 0, 0]]).T,
                              mutation_scale=5, arrowstyle="-|>", color='y', lw=1, label='photon'))
        ax.scatter([0], [0], [0], color='C0', s=10, label='sample')
        ax.add_artist(Arrow3D(*0.2*np.vstack([-self.slit_direction, self.slit_direction]).T +
                              self.analyzer_direction[:, np.newaxis],
                              mutation_scale=5, arrowstyle="-|>", color='C6', lw=1, label='slit direction'))
        ax.scatter(*self.analyzer_direction, color='C1', marker='s', s=20, label='detector')

        ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(0, 1), xlabel='x', ylabel='y', zlabel='z')
        ax.legend()


TPS45A1 = ARPESGeometry.from_angles(
    photon_direction=(-30, 0),
    slit_direction=(0, -90),  # check sign
    analyzer_direction=(90, 0),
)


def k_arpes_old(theta: npt.ArrayLike, e_photon: npt.ArrayLike,
                phi_det: float, v0: float, e_bind: float = 0.,
                theta2: npt.ArrayLike = 0., geometry: Optional[ARPESGeometry] = None):
    e_photon *= eV
    phi_det *= eV
    e_bind *= eV
    v0 *= eV
    e_electron = e_photon - e_bind - phi_det
    k = (angstrom*np.sqrt(2 * m_e * e_electron) * np.sin(theta)/hbar,
         angstrom*np.sqrt(2 * m_e * e_electron) * np.sin(theta2) * np.cos(theta)/hbar,  # noqa: E501
         angstrom*np.sqrt(2 * m_e * (e_electron * (np.cos(theta)**2 * np.cos(theta2)**2) + v0))/hbar)  # noqa: E501

    if geometry:
        k -= angstrom*(e_photon/(hbar*c))*geometry.photon_direction_sample_coords

    return k


def k_arpes(theta: npt.ArrayLike, e_photon: npt.ArrayLike, phi_det: float,
            v0: float, e_bind: npt.ArrayLike = 0., theta2: npt.ArrayLike = 0.,
            geometry: Optional[ARPESGeometry] = None,
            sample_normal_direction: Optional[npt.ArrayLike] = None):
    """Returns the parallel and perpendicular components of electronic plane
    wave exiting a crystal with inner potential `v0` at angle `theta` with an
    energy of `e_electron`"""

    v0 *= eV
    e_electron = (e_photon - e_bind - phi_det) * eV
    k = np.array([angstrom*np.sqrt(2 * m_e * e_electron) * np.sin(theta)/hbar,
         angstrom*np.sqrt(2 * m_e * e_electron) * np.sin(theta2) * np.cos(theta)/hbar,  # noqa: E501
         angstrom*np.sqrt(2 * m_e * (e_electron * (np.cos(theta)**2 * np.cos(theta2)**2) + v0))/hbar]).T  # noqa: E501

    if geometry:
        log.debug('using photon momentum correction')
        #k = (k - angstrom*(e_photon/(hbar*c))*geometry.photon_direction_sample_coords)
        k_gamma = geometry.photon_momentum_sample_coords(e_photon, sample_normal_direction)
        log.debug('k_gamma {}', k_gamma)
        k = k - k_gamma  # k_bef = k_aft - k_gamma

    return k
