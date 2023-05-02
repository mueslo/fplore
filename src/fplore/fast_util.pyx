# cython: language_level=3
# cython: infer_types=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: embedsignature=True
cimport numpy as np
from numpy.math cimport INFINITY
cimport cython
import numpy as np
from libc.math cimport sqrt
from cython.parallel cimport prange

cdef double BOUNDARY_ATOL = 1e-6

def backfold_k_inplace(double[:, ::1] neighbours_k, double[:, ::1] k_points) -> None:
    ##remove for inplace
    #cdef double[:, ::1] k_points = np.empty_like(b)
    #k_points[...] = b
    cdef unsigned char[::1] check = np.full(k_points.shape[0]//8 + 1, 0xff, dtype=np.uint8)

    cdef unsigned long nbf
    #cdef unsigned long nvisited
    cdef double mind  # minimum distance to neighbours
    cdef double d     # currently calculated distance
    cdef long imind   # index of minimum distance neighbour

    cdef long ik  # faster w/ long, b/c shape is in long
    cdef long ineigh  # faster w/ long, b/c shape is in long
    cdef long idx
    cdef char subidx

    cdef char it = 0

    with cython.nogil:
        while True:
            it += 1
            if it > 20:
                with cython.gil:
                    raise Exception('Backfolding failed')

            nbf = 0
            #nvisited = 0
            for ik in prange(k_points.shape[0], nogil=False):
                # although many openmp threads may access the same byte
                # we do not require a lock because each thread effectively
                # only accesses and modifies a single bit, and the order
                # in which it is written does not matter. yay!
                idx = ik//8  #index to char
                subidx = 1 << (ik % 8)  # index to right bit in char idx
                if not (check[idx] & subidx):
                    continue
                #nvisited += 1

                mind = INFINITY
                imind = -1
                for ineigh in range(neighbours_k.shape[0]):
                    d = sqrt((k_points[ik, 0] - neighbours_k[ineigh, 0])**2 +
                             (k_points[ik, 1] - neighbours_k[ineigh, 1])**2 +
                             (k_points[ik, 2] - neighbours_k[ineigh, 2])**2)
                    if d < mind - BOUNDARY_ATOL:
                        mind = d
                        imind = ineigh
                if imind != 0:
                    nbf += 1
                    k_points[ik, 0] -= neighbours_k[imind, 0]
                    k_points[ik, 1] -= neighbours_k[imind, 1]
                    k_points[ik, 2] -= neighbours_k[imind, 2]
                else:
                    check[idx] &= ~subidx  # remove from check

            #print('it', it, 'backfolded', nbf, 'of', nvisited, 'total:', k_points.shape[0])
            if nbf == 0:
                with cython.gil:
                    return
                #    return np.asarray(k_points)  # inplace needs no return value