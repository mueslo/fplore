# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import re

from six.moves import zip_longest
import numpy as np
from numpy.lib.recfunctions import merge_arrays
import progressbar
from scipy.stats.distributions import norm
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
from cached_property import cached_property
from pymatgen.symmetry.groups import PointGroup

from .base import FPLOFile, writeable, loads
from ..logging import log
from ..util import (cartesian_product, detect_grid, snap_to_grid,
                    remove_duplicates, in_hull)


# todo unify/subclass Band parser
# todo bweights
# todo _lms

class BandBase(object):
    @staticmethod
    def _gen_band_data_array(num_k, num_e=None,
                             ik=False, fractional_coords=False, k_coords=False,
                             weights=False, index=False,
                             **kwargs):
        ftype = kwargs.get('ftype', np.float32)
        dtype = []

        if ik:
            dtype.append(('ik', ftype))

        if fractional_coords:
            dtype.append(('frac', ftype, (3,)))

        if k_coords:
            dtype.append(('k', ftype, (3,)))

        if num_e is not None:
            dtype.append(('e', ftype, (num_e,)))

        if index:
            index_type = kwargs.get('idx_type', np.uint32)
            dtype.append(('idx', index_type))

        if weights:
            dtype.append(('c', ftype, (num_e, num_e)))

        return np.zeros(num_k, dtype=dtype)


class BandWeights(BandBase, FPLOFile):
    __fplo_file__ = ("+bweights", "+bweights_kp")

    @loads('data', 'orbitals', disk_cache=True, mem_map={'data'})
    def load(self):
        weights_file = open(self.filepath, 'r')
        header_str = next(weights_file)
        _0, _1, n_k, _3, n_bands, n_spinstates, _6, size2 = (
            f(x) for f, x in zip((int, float, int, int, int, int, int, int),
                                 header_str.split()[1:]))

        # _0: ?
        # _1: energy-related?
        # _2: number of k_points sampled
        # _3: num weights? should be equal n_bands or 0 (?)
        # _4: number of bands (1), ?
        # _5: number of spin states
        # _6: ?
        # _7: number of bands (2), ?

        columns = next(weights_file)
        columns = re.sub("[ ]{2,}", "  ", columns)
        columns = columns.split("  ")[1:-1]  # remove # and \n
        orbitals = columns[2:]

        bar = progressbar.ProgressBar(max_value=n_k*n_bands)

        data = self._gen_band_data_array(n_k, n_bands, ik=True, weights=True)

        for i, lines in bar(enumerate(
                zip_longest(*[weights_file] * n_bands))):

            e = []
            weights = []

            for line in lines:
                linedata = [float(d) for d in line.split()]
                e.append(linedata[1])
                weights.append(linedata[2:])

            data[i]['ik'] = linedata[0]
            data[i]['e'] = e
            data[i]['c'] = weights

        log.info('Band weight data is {} MiB in size',
                 data.nbytes / (1024 * 1024))

        return data, orbitals


class Band(BandBase, FPLOFile):
    __fplo_file__ = ("+band", "+band_kp")

    @loads('_data', disk_cache=True, mem_map={'_data'})
    def load(self):
        band_kp_file = open(self.filepath, 'r')
        header_str = next(band_kp_file)

        log.debug(header_str)
        _0, _1, n_k, _3, n_bands, n_spinstates, _6, size2 = (
            f(x) for f, x in zip((int, float, int, int, int, int, int, int),
                                 header_str.split()[1:]))

        bar = progressbar.ProgressBar(max_value=n_k)

        # k and e appear to be 4-byte (32 bit) floats
        data = self._gen_band_data_array(
            n_k, n_bands, ik=True, fractional_coords=True)

        for i, lines in bar(enumerate(
                zip_longest(*[band_kp_file] * (1 + n_spinstates)))):
            # read two lines at once for n_spinstates=1, three for n_s=2

            # first line
            frac = tuple(map(float, lines[0].split()[1:]))

            # second (+ third) line
            ik = float(lines[1].split()[0])
            e = [list(map(float, line_e.split()[1:])) for line_e in lines[1:]]

            # todo: don't ignore magnetic split
            e = e[0]  # ignore magnetic split

            data[i]['ik'] = ik
            data[i]['frac'] = frac
            data[i]['e'] = e

        log.info('Band data is {} MiB in size', data.nbytes / (1024 * 1024))

        return data

    @cached_property
    def data(self):
        """Returns the raw band data plus k-coordinates folded back to the
        first BZ"""

        # convert fractional coordinates to k-space coordinates
        k = self.run.frac_to_k(self._data['frac'])

        # wrap points to primitive unit cell BZ
        k = self.run.backfold_k(k)

        view_type = np.dtype([('k', k.dtype, (3,))])
        k_structured = k.view(view_type)[:, 0]

        return merge_arrays([self._data, k_structured], flatten=True)

    @cached_property
    def symm_data(self):
        """Returns the band data folded back to the first BZ and applies
        symmetry operations. Returns an index array to reduce memory usage."""
        pg = PointGroup(self.run.spacegroup.point_group)

        # apply symmetry operations from point group
        return self.apply_symmetry(self.data, pg.symmetry_ops)

    def reshape_gridded_data(self, apply_symmetries=True,
                             fractional_coords=False):
        """Tries to detect if the band data coordinates form a regular,
        rectangular grid, and returns the band data `indexes` reshaped to that
        grid."""

        # todo fractional coords
        # todo 2d reshape

        if apply_symmetries:
            data = self.symm_data
        else:
            data = self._gen_band_data_array(len(self.data),
                                             k_coords=True, index=True)
            data['k'] = self.data['k']
            data['idx'] = np.arange(len(self.data))

        if fractional_coords:
            raise NotImplementedError

        xs, ys, zs = axes = detect_grid(data['k'])
        with writeable(data):
            data['k'] = snap_to_grid(data['k'], *axes)
        regular_grid_coords = cartesian_product(*axes)
        shape = len(xs), len(ys), len(zs)

        k = data['k']
        sort_idx = np.lexsort((k[:, 2], k[:, 1], k[:, 0]))

        sorted_data = data[sort_idx]

        if np.array_equal(sorted_data['k'], regular_grid_coords):
            log.debug('detected regular k-sample grid of shape {}', shape)

            return axes, sorted_data.reshape(*shape)['idx']

        else:
            log.debug('detected sparse k-sample grid')
            # skipping check that sorted_data['k'] is a subset, in that case
            # (irregular k-points), detect_grid should throw an AssertionError

            sd_coords = np.core.records.fromarrays(
                sorted_data['k'].T, formats="f4, f4, f4")
            rgc_coords = np.core.records.fromarrays(
                regular_grid_coords.T, formats="f4, f4, f4")

            missing_coords = np.setdiff1d(rgc_coords, sd_coords,
                                          assume_unique=True)

            # due to float inaccuracy errors, sorted_data may not be a strict
            # subset of regular_grid_coords
            if len(missing_coords) != len(rgc_coords) - len(sd_coords):
                log.error("FIXME float inaccuracy errors")

            new_data = self._gen_band_data_array(
                len(regular_grid_coords), k_coords=True, index=True)

            # add existing data
            new_data[:len(sorted_data)]['k'] = sorted_data['k']
            new_data[:len(sorted_data)]['idx'] = sorted_data['idx']

            # add missing coordinates
            mc_start = len(sorted_data) - len(regular_grid_coords)
            new_data[mc_start:]['k'] = missing_coords.view('3f4')

            # backfold missing coordinates
            missing_coords = self.run.backfold_k(missing_coords.view('3f4'))

            # assert that backfolded missing coordinates are within the
            # convex hull of data present
            # todo: if not, return masked array
            missing_in_hull = in_hull(missing_coords, sorted_data['k'],
                                      tol=1e-5)
            assert missing_in_hull.all()

            # find exact matches
            all_k = new_data['k'].copy()
            all_k[mc_start:] = missing_coords
            k_u, idx_u, inv_u = np.unique(
                all_k.round(decimals=4), axis=0,
                return_index=True, return_inverse=True)

            # assert that all missing coordinates have an exact match in data
            # todo (?): if not, interpolate coordinates
            assert np.array_equal(idx_u, np.arange(len(sorted_data)))
            new_data['idx'] = new_data['idx'][inv_u]

            new_k = new_data['k']
            nsd_idx = np.lexsort((new_k[:, 2], new_k[:, 1], new_k[:, 0]))
            new_sorted_data = new_data[nsd_idx]
            assert np.array_equal(new_sorted_data['k'], regular_grid_coords)

            return axes, new_sorted_data.reshape(*shape)['idx']

    @cached_property
    def interpolator(self):
        if self.reshape_gridded_data() is None:
            log.warning('doing expensive irregular interpolation')
            return LinearNDInterpolator(self.data['k'], self.data['e'])

        axes, data = self.reshape_gridded_data()
        return RegularGridInterpolator(axes, data['e'])

    @cached_property
    def symm_interpolator(self):
        return LinearNDInterpolator(
            self.symm_data['k'], self.band['e'][self.symm_data['idx']])

    def bands_at_energy(self, e=0., tol=0.05):
        """Returns the indices of bands which cross a certain energy level"""
        idx_band = np.any(np.abs(self.data['e'] - e) < tol, axis=0)
        return np.where(idx_band)[0]

    def bands_within(self, e_lower=-0.025, e_upper=0.025):
        e = (e_lower + e_upper) / 2
        tol = np.abs(e_upper - e_lower) / 2
        return self.bands_at_energy(e=e, tol=tol)

    @staticmethod
    def smooth_overlap(e_k_3d, e=0., scale=0.02, axis=2):
        e_k_3d[np.isnan(e_k_3d)] = -np.inf
        t1 = norm.pdf(e_k_3d, loc=e, scale=scale)
        # todo interpolate axis 2

        return np.sum(t1, axis=(axis, 3))

    @classmethod
    def apply_symmetry(cls, data, symm_ops):
        """
        Applies the given symmetry operations to the k-space coordinates in
        `data` and returns a structured array with fields `k` and `idx`.

        `k` is the coordinate of the band data in k-space.
        `ìdx` is an index of `data`, where the band data equivalent to `k` can
        be found.
        """

        k_points = data['k']

        num_k, num_e = data['e'].shape
        k_idx = np.arange(num_k)
        new_data_idx = cls._gen_band_data_array(num_k*len(symm_ops),
                                                k_coords=True, index=True)
        for i, op in enumerate(symm_ops):
            rot = op.rotation_matrix
            new_k_points = np.dot(rot, k_points.T).T
            new_data_idx['k'][num_k*i:num_k*(i+1)] = new_k_points
            new_data_idx['idx'][num_k*i:num_k*(i+1)] = k_idx

        log.debug('applied {} symm ops to {} k points', len(symm_ops), num_k)
        new_data_idx = remove_duplicates(new_data_idx)

        return new_data_idx
