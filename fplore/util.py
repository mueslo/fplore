import itertools

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.distance import cdist

from .logging import log

nbs = (0, 1, -1)
neighbours = list(itertools.product(nbs, nbs, nbs))


def cartesian_product(*xs):
    """Iterates over primary axis first, then second, etc."""
    return np.array(np.meshgrid(*xs, indexing='ij')).reshape(len(xs), -1).T


def detect_grid(coordinates):
    # check if sample points form regular grid
    # normalise ordering
    # reshape
    # make interpolator

    coord_round = coordinates.round(decimals=5)
    xs = sorted(np.unique(coord_round[:, 0]))
    ys = sorted(np.unique(coord_round[:, 1]))
    zs = sorted(np.unique(coord_round[:, 2]))

    # assumption: fraction coords were laid out on regular, rectangular
    #             grid parallel to axes
    # test:
    dtype = coordinates.dtype
    xs_grid = np.linspace(xs[0], xs[-1], len(xs), dtype=dtype)
    ys_grid = np.linspace(ys[0], ys[-1], len(ys), dtype=dtype)
    zs_grid = np.linspace(zs[0], zs[-1], len(zs), dtype=dtype)
    try:
        tol = {'rtol': 0, 'atol': 1e-5}
        assert np.allclose(xs, xs_grid, **tol)
        assert np.allclose(ys, ys_grid, **tol)
        assert np.allclose(zs, zs_grid, **tol)
    except AssertionError:
        log.debug('detected irregular k-sample grid')
        log.debug("zip: {}", list(
            zip(xs, xs_grid, np.isclose(xs, xs_grid, **tol))))
        log.debug("zip: {}", list(
            zip(ys, ys_grid, np.isclose(ys, ys_grid, **tol))))
        log.debug("zip: {}", list(
            zip(zs, zs_grid, np.isclose(zs, zs_grid, **tol))))
        raise

    return xs_grid, ys_grid, zs_grid


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
    # todo specify fractional/k-space

    if energy_levels is None:
        energy_levels = np.arange(
            reshaped_data['e'].shape[-1])

    if np.isnan(reshaped_data['e']).any():
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
                reshaped_data['e'][..., i_e], idx_coords.T, order=order)
        return res

    ret = mc(energy_levels)
    ret = ret.reshape(base_shape + (len(energy_levels),))
    if np.all(np.isnan(ret)):
        raise Exception("It's all downhill from here (only NaN in result)")
    elif np.any(np.isnan(ret)):
        log.warning("NaN in sample_e output")
    return ret


def backfold_k(A, b):
    """
    Wraps an array of k-points b (shape (n_points, 3)) back to the first
    Brillouin zone given a reciprocal lattice matrix A.

    Note: Assumes that the lattice vectors contained in A correspond to the
    shortest lattice vectors, i.e. that pairwise combinations of reciprocal
    lattice vectors in A and their negatives cover all the nearest neighbours
    of the BZ.
    """

    # get adjacent BZ cell's Gamma point locations
    neighbours_k = np.dot(np.array(neighbours), A)
    idx_first_bz = np.argwhere(
        (neighbours_k == [0., 0., 0.]).all(axis=1))[0, 0]

    # make a copy of `b' since we will be operating directly on it
    b = np.copy(b)

    # all coordinates need to be backfolded initially
    idx_requires_backfolding = np.arange(len(b))
    i = 0
    while True:
        i += 1
        log.debug('backfolding... (round {})', i)

        # calculate distances to nearest neighbour BZ origins
        dists = cdist(b[idx_requires_backfolding], neighbours_k)

        # prevent float inaccuracies from folding on 1st BZ borders:
        dists[:, idx_first_bz] -= 1e-8

        # get the index of the BZ origin to which distance is minimal
        bz_idx = np.argmin(dists, axis=1)

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
            return b
        elif (np.sum(idx_backfolded)/len(idx_backfolded) < .01 or
              np.sum(idx_backfolded) < 5):
            log.debug("backfolded:")
            for i in np.argwhere(idx_backfolded):
                log.debug(
                    "  {} -> {}",
                    b[idx_requires_backfolding][i],
                    backfolded[i])

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

    return np.dot(grid, A) + start
