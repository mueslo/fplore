# -*- coding: utf-8 -*-
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from matplotlib.collections import PolyCollection, LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import colors as mcolors

from .logging import log
from .util import wigner_seitz_neighbours


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

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        if renderer is not None:  # matplotlib<3.6
            FancyArrowPatch.draw(self, renderer)
        return min(zs)

    draw = do_3d_projection  # for matplotlib<3.5


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


def plot_wigner_seitz(ax, lattice, **kwargs):
    ws_cell = lattice.get_wigner_seitz_cell()
    ax.add_collection3d(Poly3DCollection(ws_cell, **kwargs))


def _plot_bz(ax, lattice, **kwargs):
    return plot_wigner_seitz(ax, lattice.reciprocal_lattice, **kwargs)


def plot_bz(run, ax, vectors='primitive', k_points=False, use_symmetry=False,
            high_symm_points=True, rot=None, offset=(0, 0, 0)):
    if rot is None:
        rot = np.eye(3)
    offset = np.array(offset) @ run.primitive_lattice.reciprocal_lattice.matrix

    def cc(arg):
        return mcolors.to_rgba(arg, alpha=0.1)

    if k_points:
        if use_symmetry:
            points = run.band.symm_data['k']
        else:
            points = run.band.data['k']

        points = (points + offset) @ rot
        ax.plot(*points.T, marker='.', ls='',
                label='sample k-point', ms=1)

    if vectors in ('primitive', 'conventional'):
        lattice = run.primitive_lattice if vectors == 'primitive' else run.lattice
        for vec, label in zip(lattice.reciprocal_lattice.matrix, 'abc'):
            vec = (vec + offset) @ rot
            origin = offset @ rot
            ax.add_artist(Arrow3D(*zip(origin, vec), mutation_scale=20,
                                  lw=3, arrowstyle="-|>", color="r"))
            ax.text(*vec, s=label, color="r")

    facets = [[(coord + offset) @ rot for coord in facet] for facet in run.brillouin_zone]
    ax.add_collection3d(
        Poly3DCollection(facets, facecolors=cc('k'),
                         edgecolors='k'))

    if high_symm_points:
        points = run.high_symm_kpoints
        points = {k: (v + offset) @ rot for k, v in points.items()}
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


def plot_bz_proj(run, ax, neighbours=False, rot=None, axis=-1, vectors=True,
                 **kwargs):
    """Projects along given axis (default: last axis) after applying
    rotation matrix rot"""
    if rot is None:
        rot = np.eye(3)

    visible_axes = [True] * 3
    visible_axes[axis] = False

    lines = []

    if not neighbours:
        neighbours = []
    elif neighbours is True:
        neighbours = wigner_seitz_neighbours(run.primitive_lattice.reciprocal_lattice)
    else:
        neighbours = np.array(neighbours)
        neighbours = neighbours @ run.primitive_lattice.reciprocal_lattice.matrix

    for facet in run.brillouin_zone:
        facet = np.stack(facet)
        facet = np.array(list(zip(facet, np.roll(facet, -1, axis=0))))
        lines.extend(facet)
        for nb in neighbours:
            lines.extend((facet + nb))

    # project facets
    P = rot[:, visible_axes]
    lines = [facet @ P for facet in lines]

    # lines = np.array(lines)
    # todo: remove duplicate lines

    if vectors:
        x, y, z = P

        ax.arrow(0, 0, *x)
        ax.text(*x, '100')

        ax.arrow(0, 0, *y)
        ax.text(*y, '010')

        ax.arrow(0, 0, *z)
        ax.text(*z, '001')

    lines = LineCollection(lines, label='Brillouin zone', **kwargs)
    ax.add_collection(lines)
    ax.set_aspect('equal')
    ax.autoscale_view(tight=True)
