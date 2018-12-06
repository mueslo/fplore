# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import re
import itertools

import numpy as np
import progressbar
from scipy.stats.distributions import norm
from scipy.interpolate import (RegularGridInterpolator,
                               LinearNDInterpolator)
from cached_property import cached_property
from pymatgen.symmetry.groups import PointGroup

from .base import FPLOFile, writeable, cache
from ..logging import log
from ..util import (cartesian_product, detect_grid, snap_to_grid,
                    remove_duplicates)


# todo unify/subclass Band parser
# todo bweights
# todo _lms

class BandBase(object):
    @staticmethod
    def _gen_band_data_array(num_k, num_e, weights=False, ftype='float32'):
        dtype = [('ik', ftype),
                 ('k', ftype, (3,)),
                 ('e', ftype, (num_e,))]
        if weights:
            dtype.append(('c', ftype, (num_e, num_e)))

        return np.zeros(num_k, )


class BandWeights(FPLOFile):
    __fplo_file__ = ("+bweights", "+bweights_kp")

    def _load(self):
        weights_file = open(self.filepath, 'r')
        header_str = next(weights_file)
        _0, _1, n_k, _3, n_bands, n_spinstates, _6, size2 = (
            f(x) for f, x in zip((int, float, int, int, int, int, int, int),
                                 header_str.split()[1:]))

        # _0: ?
        # _1: energy-related?
        # _2: number of k_points sampled
        # _3: num weights? should be equal n_bands or 0
        # _4: number of bands (1), ?
        # _5: number of spin states
        # _6: ?
        # _7: number of bands (2), ?

        columns = next(weights_file)
        columns = re.sub("[ ]{2,}", "  ", columns)
        columns = columns.split("  ")[1:-1]  # remove # and \n
        self.orbitals = columns[2:]
        log.debug(self.orbitals)

        bar = progressbar.ProgressBar(max_value=n_k*n_bands)

        self.data = np.zeros(n_k, dtype=[
            ('ik', 'f4'),
            ('e', '{}f4'.format(n_bands)),
            ('c', '({0},{0})f4'.format(n_bands)),
        ])

        for i, lines in bar(enumerate(
                itertools.zip_longest(*[weights_file] * n_bands))):

            e = []
            weights = []

            for line in lines:
                data = [float(d) for d in line.split()]
                e.append(data[1])
                weights.append(data[2:])

            self.data[i]['ik'] = data[0]
            self.data[i]['e'] = e
            self.data[i]['c'] = weights

        log.info('Band weight data is {} MiB in size',
                 self.data.nbytes / (1024 * 1024))


class Band(FPLOFile):
    __fplo_file__ = ("+band", "+band_kp")
    # todo: custom band data classes that only store reference to band data

    @staticmethod
    def _gen_band_data_index_array(num_k, idx_type=np.uint32):
        return np.zeros(num_k, dtype=[
            ('k', '3f4'),
            ('idx', idx_type),
        ])

    @staticmethod
    def _gen_band_data_array(num_k, num_e):
        return np.zeros(num_k, dtype=[
            ('ik', 'f4'),
            ('k', '3f4'),
            ('e', 'f4', (num_e,)),
        ])

    @cache("_data")
    def _load(self):
        band_kp_file = open(self.filepath, 'r')
        header_str = next(band_kp_file)

        log.debug(header_str)
        _0, _1, n_k, _3, n_bands, n_spinstates, _6, size2 = (
            f(x) for f, x in zip((int, float, int, int, int, int, int, int),
                                 header_str.split()[1:]))

        bar = progressbar.ProgressBar(max_value=n_k)

        # k and e appear to be 4-byte (32 bit) floats
        self._data = self._gen_band_data_array(n_k, n_bands)

        for i, lines in bar(enumerate(
                itertools.zip_longest(*[band_kp_file] * (1 + n_spinstates)))):
            # read two lines at once for n_spinstates=1, three for n_s=2

            # first line
            k = tuple(map(float, lines[0].split()[1:]))

            # second (+ third) line
            ik = float(lines[1].split()[0])
            e = [list(map(float, line_e.split()[1:])) for line_e in lines[1:]]

            # todo: don't ignore magnetic split
            e = e[0]  # ignore magnetic split

            self._data[i]['ik'] = ik
            self._data[i]['k'] = k
            self._data[i]['e'] = e

        log.info('Band data is {} MiB in size',
                 self._data.nbytes / (1024 * 1024))

    def reshape(self, dimensions=None):
        shape = dimensions or self.shape()
        if shape == 1:
            return self._data

        if not self.is_rectangular_grid():
            raise Exception("can't handle uneven arrays yet")

        if shape == 2:
            return self.as_2d()
        if shape == 3:
            return self.as_3d()

    @cached_property
    def data(self):
        """Returns the band data folded back to the first BZ"""
        # convert fractional coordinates to k-space coordinates
        points = self.run.frac_to_k(self._data['k'])

        # wrap points to primitive unit cell BZ
        points = self.run.backfold_k(points)

        data = self._data.copy()
        data['k'] = points

        return data

    @cached_property
    def symm_data(self):
        """Returns the band data folded back to the first BZ and applies
        symmetry operations"""
        pg = PointGroup(self.run.spacegroup.point_group)

        # apply symmetry operations from point group
        data = np.array(self.apply_symmetry(self.data, pg.symmetry_ops))

        return data

    def reshape_gridded_data(self, apply_symmetries=True,
                             fractional_coords=False, energy_levels=None):
        if apply_symmetries:
            data = self.symm_data
        else:
            data = self.data

        if fractional_coords:
            data = np.copy(data)
            data['k'] = self.run.k_to_frac(data['k'])

        n_e = data['e'].shape[1]
        if energy_levels is None:
            energy_levels = np.arange(n_e)

        # todo: add 2d reshape ability

        xs, ys, zs = axes = detect_grid(data['k'])
        with writeable(data):
            data['k'] = snap_to_grid(data['k'], *axes)
        regular_grid_coords = cartesian_product(*axes)
        shape = len(xs), len(ys), len(zs)

        k = data['k']
        sorted_data = self._gen_band_data_array(len(data), len(energy_levels))
        sort_idx = np.lexsort((k[:, 2], k[:, 1], k[:, 0]))
        sorted_data['k'] = k[sort_idx]
        sorted_data['e'] = data['e'][..., energy_levels][sort_idx]

        if np.array_equal(sorted_data['k'], regular_grid_coords):
            log.debug('detected regular k-sample grid of shape {}', shape)

            return axes, sorted_data.reshape(*shape)

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
                len(regular_grid_coords), len(energy_levels))
            new_data[:len(sorted_data)] = sorted_data

            mc_start = len(sorted_data) - len(regular_grid_coords)
            new_data[mc_start:]['k'] = missing_coords.view('3f4')

            # try interpolating points by backfolding into 1st bz
            ip = LinearNDInterpolator(sorted_data['k'], sorted_data['e'])
            if fractional_coords:
                missing_coords_k = self.run.frac_to_k(
                    missing_coords.view('3f4'))
            else:
                missing_coords_k = missing_coords.view('3f4')

            missing_coords_k = self.run.backfold_k(missing_coords_k)

            if fractional_coords:
                missing_coords_k = self.run.k_to_frac(missing_coords_k)

            missing_coords_e = ip(missing_coords_k)

            nd_start = len(sorted_data) - len(regular_grid_coords)
            new_data[nd_start:]['e'] = missing_coords_e

            nan = np.isnan(new_data['e'])
            log.debug("{:.2f}% NaN", 100*(np.sum(nan)/np.prod(nan.shape)))

            new_k = new_data['k']
            nsd_idx = np.lexsort((new_k[:, 2], new_k[:, 1], new_k[:, 0]))
            new_sorted_data = new_data[nsd_idx]
            assert np.array_equal(new_sorted_data['k'], regular_grid_coords)

            return axes, new_sorted_data.reshape(*shape)

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
            self.symm_data['k'], self.symm_data['e'])

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
        k_points = data['k']

        num_k, num_e = data['e'].shape
        k_idx = np.arange(num_k)
        new_data_idx = cls._gen_band_data_index_array(num_k*len(symm_ops))
        for i, op in enumerate(symm_ops):
            rot = op.rotation_matrix
            new_k_points = np.dot(rot, k_points.T).T
            new_data_idx['k'][num_k*i:num_k*(i+1)] = new_k_points
            new_data_idx['idx'][num_k*i:num_k*(i+1)] = k_idx

        log.debug('applied {} symm ops to {} k points', len(symm_ops), num_k)
        new_data_idx = remove_duplicates(new_data_idx)

        new_data = cls._gen_band_data_array(len(new_data_idx), num_e)
        new_data['k'] = new_data_idx['k']
        new_data['e'] = data['e'][new_data_idx['idx']]
        return new_data
