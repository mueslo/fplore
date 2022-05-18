import itertools

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay, Voronoi
import scipy.cluster.hierarchy as hcluster
from scipy.constants import hbar, m_e, eV, angstrom, c
from pymatgen.core import Lattice

from .logging import log

nbs = (0, 1, -1)
neighbours = list(itertools.product(nbs, nbs, nbs))
idx_000 = neighbours.index((0, 0, 0))

def cartesian_product(*xs):
    """Iterates over primary axis first, then second, etc."""
    return np.array(np.meshgrid(*xs, indexing='ij')).reshape(len(xs), -1).T


def unique(coords, tol=1e-5):
    # todo test, untested, copied from 1d code
    # pre-clustering using exact uniqueness
    coords = np.unique(coords)

    # hierarchical clustering for uniqueness accounting for float inaccuracies
    xc = hcluster.fclusterdata(coords[:, np.newaxis], tol, criterion="distance")
    _, xu_idx = np.unique(xc, return_index=True)

    coords = coords[xu_idx]
    return coords


def detect_grid(coordinates):
    """
    Check if sample points form regular, rectangular grid

    :param coordinates:
    :return: (xs, ys, zs) axes of grid
    """
    dtype = coordinates.dtype
    coord_round = coordinates.round(decimals=6)
    tol = {'rtol': 0, 'atol': 1e-5}

    axes = []

    # clustering
    for coord_dim in coord_round.T:
        # pre-clustering (not really unique due to float + rounding ridges)
        xs = np.unique(coord_dim)

        # hierarchical clustering
        xc = hcluster.fclusterdata(xs[:, np.newaxis], 1e-5, criterion="distance")
        _, xu_idx = np.unique(xc, return_index=True)
        xs = sorted(xs[xu_idx])

        xs_step = np.diff(xs)
        assert np.allclose(xs_step, np.median(xs_step), **tol), "xs_step"
        axes.append(xs)

    # assumption: fraction coords were laid out on regular, rectangular
    #             grid parallel to axes
    # test:

    g_min = np.min(coordinates, axis=0)
    g_max = np.max(coordinates, axis=0)

    axes_grid = []
    for dim_min, dim_max, xs in zip(g_min.T, g_max.T, axes):
        xs_grid = np.linspace(dim_min, dim_max, len(xs), dtype=dtype)
        assert np.allclose(xs, xs_grid, **tol), "xs"
        axes_grid.append(xs_grid)

    return axes_grid


def find_basis(lattice_points):  
    basis = []
    dists = np.linalg.norm(lattice_points, axis=1)
    valid = (dists != 0)
    lattice_points = lattice_points[valid]
    dists = dists[valid]
    
    n=12
    
    # find n shortest vectors
    partition = np.argpartition(dists, n)[:n]
    order = np.argsort(dists[partition])
    smallest = lattice_points[partition][order]
    
    basis = None
    for basis_candidate in itertools.permutations(smallest, 3):
        if np.linalg.matrix_rank(basis_candidate) < 3:
            continue
        basis = np.array(basis_candidate) 
        break
    else:
        raise Exception('Points do not span a volume')

    try:
        # diff to grid
        inv_basis = np.linalg.inv(basis)
        frac = lattice_points @ inv_basis
        diff = frac - np.rint(frac)
        max_diff = np.linalg.norm(diff, axis=1).max()
        assert max_diff < 1e-4, f'max_diff {max_diff}'

        # assert grid has no major holes (e.g. basis too small)
        uniq_x, uniq_y, uniq_z = [np.unique(frac[:, i].round(decimals=4)) for i in range(3)]
        assert np.allclose(uniq_x, np.arange(uniq_x[0], uniq_x[-1]+1)), 'x'
        assert np.allclose(uniq_y, np.arange(uniq_y[0], uniq_y[-1]+1)), 'y'
        assert np.allclose(uniq_z, np.arange(uniq_z[0], uniq_z[-1]+1)), 'z'
    except AssertionError as ae:
        log.debug('basis {}\n{}', basis, ae)
        raise ValueError('Could not determine basis from input lattice_points')

    return basis


def find_lattice(lattice_points):
    return Lattice(find_basis(lattice_points)).get_niggli_reduced_lattice()


def snap_to_grid(points, *grid_axes):
    snapped = []
    for i, ax in enumerate(grid_axes):
        diff = ax[:, np.newaxis] - points[:, i]
        best = np.argmin(np.abs(diff), axis=0)
        snapped.append(ax[best])
    return np.array(snapped).T


def sample_e(axes, reshaped_data, coords, order=1,
             energy_levels=None):
    """Sample a given ndimage (axes, reshaped_data) at coords"""

    if energy_levels is None:
        energy_levels = np.arange(
            reshaped_data.shape[-1])

    if np.isnan(reshaped_data).any():
        log.warning("NaN in sample_e input")

    len_axes = np.array([len(axis) for axis in axes])
    axes = np.array([[axis[0], axis[-1]] for axis in axes])
    # x0 + ix/(nx - 1) * (x_end - x0) = coord
    # -> (nx - 1) * (coord - x0)/(x_end - x0) = ix
    idx_coords = ((len_axes - 1) *
                  (coords - axes[:, 0]) / (axes[:, -1] - axes[:, 0]))

    assert coords.shape[-1] == 3
    base_shape = coords.shape[:-1]
    if len(coords.shape) > 2:
        idx_coords = idx_coords.reshape(-1, 3)
        coords = coords.reshape(-1, 3)

    # since scipy.ndimage.map_coordinates cannot handle vector-valued
    # functions, iterate over energy indices
    def mc(i_es):
        res = np.zeros((len(coords), len(i_es)))
        for i, i_e in enumerate(i_es):
            res[:, i] = map_coordinates(
                reshaped_data[..., i_e], idx_coords.T, order=order)
        return res

    ret = mc(energy_levels)
    ret = ret.reshape(base_shape + (len(energy_levels),))
    if np.all(np.isnan(ret)):
        raise Exception("It's all downhill from here (only NaN in result)")
    elif np.any(np.isnan(ret)):
        log.warning("NaN in sample_e output")
    return ret


def wigner_seitz_neighbours(lattice):
    ksamp_lattice_points = np.array(neighbours) @ lattice.matrix
    vor = Voronoi(ksamp_lattice_points)

    ws_nbs = frozenset(itertools.chain.from_iterable(
        set(a) for a in vor.ridge_points if idx_000 in a))  - {idx_000}
    ws_nbs = np.array(list(ws_nbs))

    ws_nbs = vor.points[ws_nbs] @ lattice.inv_matrix
    ws_nbs = set(map(tuple, np.rint(ws_nbs).astype(int)))

    return ws_nbs  # wigner-seitz cell neighbours in units of basis vectors


def fill_bz(k, reciprocal_lattice, ksamp_lattice=None, pad=False):
    ksamp_lattice = ksamp_lattice or find_lattice(k)
    neighbours_lattice = wigner_seitz_neighbours(ksamp_lattice)

    #idx_first_bz = in_first_bz(k, reciprocal_lattice)
    #k = k[idx_first_bz]

    ksamp_lattice_ijk = k @ ksamp_lattice.inv_matrix  # (N, 3) array of float
    ksamp_lattice_ijk = set(map(tuple, np.rint(ksamp_lattice_ijk).astype(int)))  # convert to set of tuples of int
    checked = set()
    to_check = ksamp_lattice_ijk.copy()

    while to_check:
        new_ijk = set()
        for p in to_check:
            new_ijk.update(set((p[0] + nb[0], p[1] + nb[1], p[2] + nb[2]) for nb in neighbours_lattice))
        
        checked |= to_check
        new_ijk -= checked

        if pad:
            checked |= new_ijk
        new_ijk = np.array(list(new_ijk))
        new_ijk_in1bz = in_first_bz(new_ijk @ ksamp_lattice.matrix, reciprocal_lattice)
        new_ijk = new_ijk[new_ijk_in1bz] # remove those not inside 1st bz
        to_check = set(map(tuple, new_ijk))
    
    return np.array(list(checked - ksamp_lattice_ijk)) @ ksamp_lattice.matrix

"""
def pad_regular_sampling_lattice(k, ksamp_lattice=None):
    ksamp_lattice = ksamp_lattice or find_lattice(k)
    neighbours_lattice = wigner_seitz_neighbours(ksamp_lattice)

    ksamp_lattice_ijk = k @ ksamp_lattice.inv_matrix  # (N, 3) array of float
    ksamp_lattice_ijk = set(map(tuple, np.rint(ksamp_lattice_ijk).astype(int)))  # convert to set of tuples of int

    expanded_lattice_ijk = ksamp_lattice_ijk.copy()
    for p in ksamp_lattice_ijk:
        expanded_lattice_ijk.update(set((p[0] + nb[0], p[1] + nb[1], p[2] + nb[2]) for nb in neighbours_lattice))

    extra_ijk = np.array(list(expanded_lattice_ijk - ksamp_lattice_ijk))
    extra_k = extra_ijk @ ksamp_lattice.matrix

    return extra_k, extra_ijk"""


def backfold_k_parallelepiped(lattice, b, atol=1e-4):
    b_shape = b.shape
    b = b.reshape((-1, 3))  # convert (..., 3) to (N, 3) shape
    b = ((b @ lattice.inv_matrix + atol) % 1 - atol) @ lattice.matrix
    return b.reshape(b_shape)


def backfold_k(lattice, b):
    """
    Folds an array of k-points b (shape (..., 3)) back to the first
    Brillouin zone given a reciprocal lattice matrix A.

    Translationally equivalent points will be mapped to the same actual point.

    Note: Assumes that the lattice vectors contained in A correspond to the
    shortest lattice vectors, i.e. that pairwise combinations of reciprocal
    lattice vectors in A and their negatives cover all the nearest neighbours
    of the BZ.
    """

    # TODO handle points on borders of BZ more elegantly
    # TODO make sure translationally equivalent points not present (brillouin zone boundary)

    # get adjacent BZ cell's Gamma point locations
    assert idx_000 == 0
    neighbours_k = np.array(neighbours) @ lattice.matrix

    #would reduce memory usage:
    #neighbours_k = np.array(list(wigner_seitz_neighbours(lattice))) @ lattice.matrix
    #neighbours_k = np.vstack([[0, 0, 0], neighbours_k])

    # make a copy of `b' since we will be operating directly on it
    b_shape = b.shape
    b = np.copy(b).reshape((-1, 3))  # convert (..., 3) to (N, 3) shape

    # to reduce problems due to translational equivalence (borders of BZ)
    # we directly fold to the parallelepiped spanned by the reciprocal lattice basis
    b = backfold_k_parallelepiped(lattice, b)

    # all coordinates need to be backfolded initially
    idx_requires_backfolding = np.arange(len(b))
    i = 0
    while True:
        i += 1
        if i > 20:
            raise Exception('Backfolding failed')
        log.debug('backfolding... (round {})', i)

        # calculate distances to nearest neighbour BZ origins
        dists = cdist(b[idx_requires_backfolding], neighbours_k)

        # get the index of the BZ origin to which distance is minimal
        # bz_idx = np.argmin(dists, axis=1)  # naive, but does not work reliably w.r.t.
        #                                      translational equivalence due to float inaccuracies
        bz_idx = np.argmax(np.isclose(dists, np.min(dists, axis=1)[:, np.newaxis]), axis=1)  # argmax for first True

        # perform backfolding
        backfolded = b[idx_requires_backfolding] - neighbours_k[bz_idx]

        # get indices of points that were backfolded (boolean index array)
        idx_backfolded = np.any(b[idx_requires_backfolding] != backfolded,
                                axis=1)

        log.debug('backfolded {} of {} coordinates',
                  np.sum(idx_backfolded),
                  len(idx_requires_backfolding))

        if not np.any(idx_backfolded):
            log.debug("backfolding finished")
            return b.reshape(b_shape)

        # assign backfolded coordinates to output array
        b[idx_requires_backfolding] = backfolded

        # only those coordinates which were changed in this round need to be
        # backfolded again
        idx_requires_backfolding = idx_requires_backfolding[idx_backfolded]


def remove_duplicates(data):
    """Remove non-unique k-points from data"""

    unique, idx = np.unique(data['k'].round(decimals=5),
                            return_index=True, axis=0)

    log.debug('deleting {} duplicates from band data', len(data) - len(idx))

    data = data[idx]
    return data


def linspace_ng(start, *stops, **kwargs):
    """
    Return evenly spaced coordinates between n arbitrary 3d points

    A single stop will return a line between start and stop, two stops make
    a plane segment, three stops will span a parallelepiped.

    """
    start = np.array(start)
    stops = [np.array(stop) for stop in stops]

    num = kwargs.get("num", 50)
    vecs = [stop - start for stop in stops]

    try:
        if len(num) != len(stops):
            raise Exception("Length of 'num' and 'stops' must match")
    except TypeError:  # no len
        num = len(stops) * (num,)

    grid = np.array(np.meshgrid(
        *(np.linspace(0, 1, num=n) for n in num))).T
    A = np.vstack(vecs)

    return grid @ A + start


def in_first_bz(p, reciprocal_lattice):
    """
    Test if points `p` are in the first Brillouin zone
    """
    ws_neighbours = list(wigner_seitz_neighbours(reciprocal_lattice))
    idx_000 = 0
    ws_neighbours.insert(idx_000, (0, 0, 0))
    neighbours_k = np.array(ws_neighbours) @ reciprocal_lattice.matrix
    dists = cdist(p, neighbours_k)

    # prevent float inaccuracies from folding on 1st BZ borders:
    dists[:, idx_000] -= 1e-6

    return np.argmin(dists, axis=1) == idx_000


def in_hull(p, hull, **kwargs):
    """
    Test if points in `p` are within `hull`
    """
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p, **kwargs) >= 0


def generate_irreducible_wedge(lattice):
    """
    Partitions BZ into irreducible wedges.
    :return: List of irreducible wedges with matrix to transform coordinates
    to the 'primary' wedge.
    """
    raise NotImplementedError


def rot_v1_v2(v1, v2):
    """Returns rotation matrix which rotates v1 onto v2"""
    v1 = np.array(v1)
    v2 = np.array(v2)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    cp = np.cross(v1, v2)
    c = np.dot(v1, v2)  # cosine

    cpm = np.array([[0, -cp[2], cp[1]],
                   [cp[2], 0, -cp[0]],
                   [-cp[1], cp[0], 0]])

    return np.eye(3) + cpm + np.dot(cpm, cpm) * 1 / (1 + c)


def k_arpes(theta, e_photon, phi_det, v0, e_bind=0., theta2=0., geometry=None):
    """Returns the parallel and perpendicular components of electronic plane
    wave exiting a crystal with inner potential `v0` at angle `theta` with an
    energy of `e_electron`"""

    e_photon *= eV
    phi_det *= eV
    e_bind *= eV
    v0 *= eV
    e_electron = e_photon - e_bind - phi_det
    k = np.array([angstrom*np.sqrt(2 * m_e * e_electron) * np.sin(theta)/hbar,
         angstrom*np.sqrt(2 * m_e * e_electron) * np.sin(theta2) * np.cos(theta)/hbar,  # noqa: E501
         angstrom*np.sqrt(2 * m_e * (e_electron * (np.cos(theta)**2 * np.cos(theta2)**2) + v0))/hbar]).T  # noqa: E501

    if geometry:
        log.debug('using photon momentum correction')
        k = (k - angstrom*(e_photon/(hbar*c))*geometry.photon_direction_sample_coords)

    return k

#k_arpes = np.vectorize(k_arpes, signature='(1),(1),(1),(1),(1?),(1?),(1?)->(3)')