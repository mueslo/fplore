#!/usr/bin/env python2


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
#from scipy.spatial import Delaunay

from matplotlib.patches import Polygon
from matplotlib.collections import PolyCollection, PatchCollection

#from http://stackoverflow.com/questions/23840756/how-to-disable-perspective-in-mplot3d
def orthogonal_proj(zfront, zback):
    a = (zfront+zback)/(zfront-zback)
    b = -2*(zfront*zback)/(zfront-zback)
    return np.array([[1,0,0,0],
                      [0,1,0,0],
                      [0,0,a,b],
                      [0,0,-1e-4,zback]])

def projected_area(xyz,axis=0):
    x = xyz[..., 1-axis]
    e = xyz[..., 2]

    if axis == 1: # project along y
        y1 = e[..., 3]-e[..., 0]
        y2 = e[..., 2]-e[..., 1]
    else: # project along x
        y1 = e[..., 1]-e[..., 0]
        y2 = e[..., 2]-e[..., 3]
    dx = x[..., 2] - x[..., 0]

    eq_idx = (np.sign(y1) == np.sign(y2))

    with np.errstate(invalid='ignore'):
        ret =  np.abs(.5*dx*((y1+y2)*eq_idx +
                             (y1**2+y2**2)/(y1-y2)*(1-eq_idx)) )
        ret[np.isnan(ret)] = 0. # we get NaN when y1 == y2 == 0
        # TODO: think of more elegant solution to avoid NaN  

    return ret

def project(x, y, z, axis=0):
    #TODO project along arbitrary vectors
    # for that we need to know the total area of a self-intersecting quadrilateral, which is ???
    """
    x, y: MxN (meshgrid)
    z: MxN
    axis: int
    """
    #cyclical iteration:
    #cyc_squ[1, 0] = 
    # x -> (2nd axis) (because meshgrid swaps axes)
    #o - 0 - 1 - o y (1st axis)
    #|   |   |   | |
    #o - 3 - 2 - o v
    #|   |   |   |
    #o - o - o - o
    #|   |   |   |
    #o - o - o - o

    p = np.stack([x,y,z]).transpose(1,2,0) # (xyz),i,j -> i,j,(xyz)

    #                         0          +dx        +dy        -dx
    cyc_squ = np.stack([p[:-1, :-1], p[:-1, 1:], p[1:, 1:], p[1:, :-1]]).transpose((1,2,0,3))
    # cyc squ: idx 0, 1: M-1 x N-1 cooridnates to circular path
    #          idx 2:    4 circular path
    #          idx 3:    3 (xyz) coordinates

    print "Generating", np.prod(cyc_squ.shape[:2]), "polygons"

    pc = list() # polygons (quadrilaterals)
    fcs = list() # their colors and translucency
		                 #x                     x
    xs = cyc_squ[:, :, 2, 0] - cyc_squ[:, :, 0, 0]
                         #y                     y
    ys = cyc_squ[:, :, 2, 1] - cyc_squ[:, :, 0, 1]
    areas = xs*ys # for a rectilinear grid
                                                #e/z
    proj_areas = projected_area(cyc_squ, axis=axis)

    with np.errstate(divide='ignore'):
        alphas = 0.1 * areas/proj_areas
        alphas[alphas > 1] = 1.
    #print areas.shape, cyc_squ.shape
    #yc_squ = np.stack([cyc_squ, areas], axis=0)

    #                   x     y     z
    idx_visible_axes = [True, True, True]
    idx_visible_axes[axis] = False

    # TODO remove loop
    for iix, ix in enumerate(cyc_squ):
        for iiy, iy in enumerate(ix):
	        #alpha = min(1, np.min(areas)/area)
	        #alpha = np.min(areas)/area
	        alpha = alphas[iix, iiy]
	        #print iy, area, alpha
	        #polygon = Polygon(iy[:, 1:], True, fc='k', lw=0)
	        pc.append(iy[:, idx_visible_axes])
	        fcs.append((0, 0, 0, alpha))

    pc = PolyCollection(pc, facecolors=fcs, rasterized=True)
    return pc


if __name__ == "__main__":
    fig = plt.figure(figsize=plt.figaspect(1/3.))
    x_scale = np.linspace(0, 0.3*np.pi/2, 100)
    y_scale = np.linspace(0, np.pi/2, 250)

    axis_labels=("k_x", "k_y", "E")
    x, y = np.meshgrid(x_scale, y_scale)
    z = np.sin(x+y)

    proj_axis=1
    pc = project(x, y, z, axis=proj_axis)
    ax = fig.add_subplot(1, 3, 1)
    ax.add_collection(pc)

    plot_x_range = [x_scale,y_scale][1-proj_axis].take((0, -1))
    ax.set_xlim(plot_x_range)
    ax.set_ylim([0, 1.1])
    ax.set_xlabel(axis_labels[1-proj_axis])
    ax.set_ylabel(axis_labels[2])


    ax2 = fig.add_subplot(1, 3, 2, projection='3d')

    proj3d.persp_transformation = orthogonal_proj
    ax2.plot_surface(x,y,z, shade=True, alpha=0.9, linewidth=0)
    ax2.set_xlabel(axis_labels[0])
    ax2.set_ylabel(axis_labels[1])
    ax2.set_zlabel(axis_labels[2])

plt.show()
