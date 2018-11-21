import numpy as np
from matplotlib.collections import PolyCollection

from . import log


# from http://stackoverflow.com/questions/23840756/how-to-disable-perspective-in-mplot3d
def orthogonal_proj(zfront, zback):
    a = (zfront+zback)/(zfront-zback)
    b = -2*(zfront*zback)/(zfront-zback)
    return np.array([[1,0,0,0],
                      [0,1,0,0],
                      [0,0,a,b],
                      [0,0,-1e-4,zback]])


def projected_area(xyz, axis):
    x = xyz[..., 1-axis]
    e = xyz[..., 2]

    # 0: orig, 1: +dx, 2: +dy, 3: -dx
    if axis == 1: # project along y
        y1 = e[..., 3]-e[..., 0]
        y2 = e[..., 2]-e[..., 1]
    else: # project along x
        y1 = e[..., 1]-e[..., 0]
        y2 = e[..., 2]-e[..., 3]
    dx = x[..., 2] - x[..., 0]

    eq_idx = (np.sign(y1) == np.sign(y2))

    with np.errstate(invalid='ignore'):
        p1 = eq_idx * (y1 + y2)  # non-intersecting quadrilateral
        p2 = (1 - eq_idx) * (y1**2 + y2**2) / (y1 - y2)  # self-intersecting quadrilateral

        p2[np.logical_and(np.isnan(p2), eq_idx)] = 0.  # fix y1 = y2

        ret = np.abs(.5 * dx * (p1 + p2))

    return ret


def make_quadrilaterals(x, y, z):
    #cyclical iteration:
    #cyc_squ[1, 0] =
    # y -> (2nd axis)
    #o - 0 - 3 - o x (1st axis)
    #|   |   |   | |
    #o - 1 - 2 - o v
    #|   |   |   |
    #o - o - o - o
    #|   |   |   |
    #o - o - o - o

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
    areas = xs*ys  # for a rectilinear grid
                                #e/z
    proj_areas = projected_area(quads, axis=axis)

    with np.errstate(divide='ignore'):
        alphas = 0.1 * areas/proj_areas
        alphas[alphas > 1] = 1.

    idx_visible_axes = [True, True, True]
    idx_visible_axes[axis] = False

    pc = quads[..., idx_visible_axes]
    fcs = np.zeros((len(alphas), 4), dtype=np.float32)
    fcs[:, :3] = r, g, b
    fcs[:, 3] = a*alphas

    pc = PolyCollection(pc, facecolors=fcs, rasterized=True)
    return pc
