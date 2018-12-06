# -*- coding: utf-8 -*-
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.collections import PolyCollection, LineCollection

from .logging import log


def projected_area(xyz, axis):
    x = xyz[..., 1 - axis]
    e = xyz[..., 2]

    # 0: orig, 1: +dx, 2: +dy, 3: -dx
    if axis == 1:  # project along y
        y1 = e[..., 3] - e[..., 0]
        y2 = e[..., 2] - e[..., 1]
    else:  # project along x
        y1 = e[..., 1] - e[..., 0]
        y2 = e[..., 2] - e[..., 3]
    dx = x[..., 2] - x[..., 0]

    eq_idx = (np.sign(y1) == np.sign(y2))

    with np.errstate(invalid='ignore'):
        p1 = eq_idx * (y1 + y2)  # non-intersecting quadrilateral
        p2 = (1 - eq_idx) * (y1 ** 2 + y2 ** 2) / (
                    y1 - y2)  # self-intersecting quadrilateral

        p2[np.logical_and(np.isnan(p2), eq_idx)] = 0.  # fix y1 = y2

        ret = np.abs(.5 * dx * (p1 + p2))

    return ret


def make_quadrilaterals(x, y, z):
    # cyclical iteration:
    # cyc_squ[1, 0] =
    # y -> (2nd axis)
    # o - 0 - 3 - o x (1st axis)
    # |   |   |   | |
    # o - 1 - 2 - o v
    # |   |   |   |
    # o - o - o - o
    # |   |   |   |
    # o - o - o - o

    p = np.stack([x, y, z]).transpose(1, 2, 0)  # 3,n_x,n_y -> n_x,n_y,3

    cyc_squ = np.stack(
        #  0            +dx         +dy        -dx
        [p[:-1, :-1], p[1:, :-1], p[1:, 1:], p[:-1, 1:]]).transpose(
        (1, 2, 0, 3))
    # cyc squ: idx 0, 1: M-1 x N-1, coordinates to circular paths
    #          idx 2:    4, quadrilateral points
    #          idx 3:    3, x/y/z

    log.debug("Generating {} polygons", np.prod(cyc_squ.shape[:2]))

    return cyc_squ.reshape(-1, 4, 3)


def project(x, y, z, axis=1, color=(0., 0., 0., 1.)):
    """
    Projects z(x,y) along an axis.
    Useful for example for showing bulk states in slab calculations.
    x, y: MxN (meshgrid with 'ij' indexing)
    z: MxN
    axis: int (axis along which to project)

    Returns:
    polycollection
    """
    assert x.shape == y.shape == z.shape
    if np.isnan(z).any():
        log.warning("NaN values in projection input")
    r, g, b, a = color

    quads = make_quadrilaterals(x, y, z)

    #                x                x
    xs = quads[:, 2, 0] - quads[:, 0, 0]
    #                y                y
    ys = quads[:, 2, 1] - quads[:, 0, 1]
    areas = xs * ys  # for a rectilinear grid
    # e/z
    proj_areas = projected_area(quads, axis=axis)

    with np.errstate(divide='ignore'):
        alphas = 0.1 * areas / proj_areas
        alphas[alphas > 1] = 1.

    idx_visible_axes = [True, True, True]
    idx_visible_axes[axis] = False

    pc = quads[..., idx_visible_axes]
    fcs = np.zeros((len(alphas), 4), dtype=np.float32)
    fcs[:, :3] = r, g, b
    fcs[:, 3] = a * alphas

    pc = PolyCollection(pc, facecolors=fcs, rasterized=True)
    return pc


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


# from http://stackoverflow.com/questions/23840756/
def orthogonal_proj(zfront, zback):
    a = (zfront + zback) / (zfront - zback)
    b = -2 * (zfront * zback) / (zfront - zback)
    return np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, a, b],
                     [0, 0, -1e-4, zback]])


def plot_structure(run, ax):
    raise NotImplementedError


def plot_bz(run, ax, vectors=True, k_points=False, use_symmetry=False,
            high_symm_points=True):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib import colors as mcolors

    def cc(arg):
        return mcolors.to_rgba(arg, alpha=0.1)

    if k_points:
        if use_symmetry:
            points = run.band.symm_data['k']
        else:
            points = run.band.data['k']

        ax.plot(*points.T, marker='.', ls='',
                label='sample k-point', ms=1)

    if vectors:
        for vec in run.primitive_lattice.reciprocal_lattice.matrix:
            ax.add_artist(Arrow3D(*zip((0, 0, 0), vec), mutation_scale=20,
                                  lw=3, arrowstyle="-|>", color="r"))

    ax.add_collection3d(
        Poly3DCollection(run.brillouin_zone, facecolors=cc('k'),
                         edgecolors='k'))

    if high_symm_points:
        points = run.high_symm_kpoints
        ax.plot(*zip(*points.values()), marker='o', ls='',
                label='high symmetry point', color='k', ms='1')

        for kpath in run.high_symm_kpaths:
            path = [points[lbl] for lbl in kpath]
            ax.plot(*zip(*path), ls='-', color='k', alpha=0.5)

        for label, coord in points.items():
            ax.text(*coord, s='${}$'.format(label), color='k')

    ax.set_xlabel('$k_x$')
    ax.set_ylabel('$k_y$')
    ax.set_zlabel('$k_z$')
    ax.legend()


def plot_bz_proj(run, ax, axis=2, **kwargs):
    visible_axes = [True] * 3
    visible_axes[axis] = False
    lines = []

    for facet in run.brillouin_zone:
        proj_facet = np.stack(facet)[:, visible_axes]
        lines.append(proj_facet.tolist())

    lines = LineCollection(lines, label='Brillouin zone', **kwargs)
    ax.add_collection(lines)
    ax.autoscale_view()
