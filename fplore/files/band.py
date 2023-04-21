# -*- coding: utf-8 -*-

import re
from itertools import zip_longest
from functools import cached_property

import numpy as np
from numpy.lib.recfunctions import merge_arrays
import progressbar
from scipy.stats.distributions import norm
from scipy.interpolate import RegularGridInterpolator, LinearNDInterpolator
from pymatgen.core import Lattice

from .base import FPLOFile, writeable, loads
from ..logging import log
from ..util import (cartesian_product, find_basis, detect_grid,
                    remove_duplicates, snap_to_grid, fill_bz,
                    find_lattice)


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
            if isinstance(weights, bool):
                num_weights = num_e
            else:
                num_weights = weights
            dtype.append(('c', ftype, (num_e, num_weights)))

        return np.zeros(num_k, dtype=dtype)

    def bands_at_energy(self, e=0., tol=0.05):
        """Returns the indices of bands which cross a certain energy level"""
        idx_band = np.any(np.abs(self.data['e'] - e) < tol, axis=0)
        return np.where(idx_band)[0]

    def bands_within(self, e_lower=-0.025, e_upper=0.025):
        e = (e_lower + e_upper) / 2
        tol = np.abs(e_upper - e_lower) / 2
        return self.bands_at_energy(e=e, tol=tol)


class BandWeights(BandBase, FPLOFile):
    __fplo_file__ = re.compile(r"\+bw(eights(lms)?|sum)(_kp)?(_unfold)?")

    @loads('data', 'labels', disk_cache=True, mem_map={'data'})
    def load(self):
        weights_file = open(self.filepath, 'r')
        header_str = next(weights_file)
        _0, _1, n_k, n_weights, n_bands, n_spinstates, _6, size2 = (
            f(x) for f, x in zip((int, float, int, int, int, int, int, int),
                                 header_str.split()[1:]))

        # _0: ?
        # _1: energy-related?
        # _2: number of k_points sampled
        # _3: number of weights, should be greater or equal n_bands or 0 (?)
        # _4: number of bands (1), ?
        # _5: number of spin states
        # _6: ?
        # _7: number of bands (2), ?

        columns = next(weights_file)
        columns = re.sub("[ ]{1,}", " ", columns)
        columns = columns.split(" ")[1:-1]  # remove # and \n
        labels = columns[2:]

        bar = progressbar.ProgressBar(max_value=n_k*n_bands)

        data = self._gen_band_data_array(n_k, n_bands, ik=True,
                                         weights=n_weights)

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

        return data, labels

    @property
    def orbitals(self):
        label_re = re.compile(
            r"(?P<element>[A-Z][a-z]?)"
            r"\((?P<site>\d{3})\)"
            r"(?P<orbital>(?P<n>\d)(?P<l>[spdfg])"
            r"((?P<j>[\d/]+)(?P<mj>[+-][\d/]+)"
            r"|"
            r"(?P<ml>[+-][\d/]+)(?P<s>up|dn)))")
        return [re.fullmatch(label_re, label) or label for label in self.labels]


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
            if n_spinstates > 1:
                log.warning('Ignoring magnetic split (TODO: fix)')

            e = e[0]  # ignore magnetic split

            data[i]['ik'] = ik
            data[i]['frac'] = frac
            data[i]['e'] = e

        log.info('Band data is {} MiB in size', data.nbytes / (1024 * 1024))

        return data

    @property
    def weights(self):
        return self.run.band_weights

    @cached_property
    def data(self):
        """Returns the raw band data plus k-coordinates (if possible)"""
        if self.run is None:
            log.info('No associated FPLORun, cannot convert fractional FPLO units (2pi/a) to k')
            return self._data

        # convert fractional coordinates to k-space coordinates
        k = self.run.fplo_to_k(self._data['frac'])

        view_type = np.dtype([('k', k.dtype, (3,))])
        k_structured = k.view(view_type)[:, 0]

        return merge_arrays([self._data, k_structured], flatten=True)

    @cached_property
    def symm_data(self):
        """Returns the band data folded back to the first BZ and applies
        symmetry operations. Returns an index array to reduce memory usage."""

        # get symmetry operations from point group, in rotated cartesian coords
        symm_ops = self.run.point_group_operations

        # apply symmetry operations from point group
        data = self.apply_symmetry(self.data, symm_ops)
        data['k'] = self.run.backfold_k(data['k'])
        data = remove_duplicates(data)
        return data

    @cached_property
    def padded_symm_data(self):
        k = self.symm_data['k']
        ksamp_lattice = find_lattice(k)
        extra_k = fill_bz(k, self.run.primitive_lattice.reciprocal_lattice, ksamp_lattice=ksamp_lattice, pad=True)
        #extra_k, extra_ijk = pad_regular_sampling_lattice(k_fill, ksamp_lattice=ksamp_lattice)
        extra_k_frac = (extra_k @ self.run.primitive_lattice.reciprocal_lattice.inv_matrix + 1e-4) % 1 -1e-4 # k to reciprocal lattice vectors parallelepiped
        # 1e-4 for consistency even with float inaccuracies

        lattice_ijk = extra_k_frac @ self.run.primitive_lattice.reciprocal_lattice.matrix @ ksamp_lattice.inv_matrix
        lattice_ijk = list(map(tuple, np.rint(lattice_ijk).astype(int)))  # == extra_ijk folded back to parallelepiped, todo check residuals

        new_data = self._gen_band_data_array(len(self.symm_data) + len(extra_k),
                                             k_coords=True, index=True)

        new_data[:len(self.symm_data)] = self.symm_data
        new_data[len(self.symm_data):]['k'] = extra_k

        idx_map = self.ksamp_idx_map(ksamp_lattice)
        try:
            new_data[len(self.symm_data):]['idx'] = [idx_map[ijk] for ijk in lattice_ijk]
        except KeyError as ke:
            print(ke)
            breakpoint()

        return new_data


    def ksamp_idx_map(self, ksamp_lattice):
        """dict : ijk k-sample lattice coordinates in basis parallelepiped -> unique idx"""
        data = self.symm_data
        frac = (data['k'] @ self.run.primitive_lattice.reciprocal_lattice.inv_matrix +1e-4) % 1 -1e-4  # k to reciprocal lattice vectors parallelepiped

        # lattice_ijk maps k sample ijk values to unique index
        lattice_ijk = frac @ self.run.primitive_lattice.reciprocal_lattice.matrix @ ksamp_lattice.inv_matrix
        lattice_ijk = map(tuple, np.rint(lattice_ijk).astype(int))

        return dict(zip(lattice_ijk, data['idx']))  # map sampling lattice ijk to unique index

    def reshape_gridded_data(self, data='padded_symm', missing_coords_strategy='backfold'):
        """Tries to detect if the band data coordinates form a regular,
        rectangular grid, and returns the band data `indexes` reshaped to that
        grid."""

        if isinstance(data, np.ndarray):
            pass
        elif data in ('padded_symm', None):
            data = self.padded_symm_data
        elif data == 'symm':
            data = self.symm_data
        elif data in ('raw', 'data'):
            data = self._gen_band_data_array(len(self.data),
                                             k_coords=True, index=True)
            data['k'] = self.data['k']
            data['idx'] = np.arange(len(self.data))

        data = remove_duplicates(data)  # remove duplicate k values

        basis = find_basis(data['k'])
        if basis is None:
            log.warning('No regular k grid detected')
            return None

        lattice = Lattice(basis).get_niggli_reduced_lattice()

        if not lattice.is_orthogonal:
            log.warning('Non-orthogonal grid detected, reshape_gridded_data will return `None`')
            return None

        if np.logical_not(np.isclose(lattice.matrix, 0)).sum() > 3:
            log.debug(lattice)
            log.warning('Rotated orthogonal grid not implemented')
            return None

        xs, ys, zs = axes = detect_grid(data['k'])
        with writeable(data):
            data['k'] = snap_to_grid(data['k'], *axes)  # required to prevent float inaccuracy errors below
        regular_grid_coords = cartesian_product(*axes)
        shape = len(xs), len(ys), len(zs)

        k = data['k']
        k_set = set(map(tuple, k.round(decimals=4)))  # todo use ksamp lattice ijk
        rgc_set = set(map(tuple, regular_grid_coords.round(decimals=4)))  # todo use ksamp lattice ijk

        if k_set == rgc_set:
            log.debug('detected regular k-sample grid of shape {}', shape)
            sort_idx = np.lexsort((k[:, 2], k[:, 1], k[:, 0]))
            return axes, data[sort_idx].reshape(*shape)['idx']

        else:
            log.debug('detected sparse k-sample grid')
            # skipping check that sorted_data['k'] is a subset, in that case
            # (irregular k-points), detect_grid should throw an AssertionError

            sd_coords = np.core.records.fromarrays(
                k.T, formats="f4, f4, f4")
            rgc_coords = np.core.records.fromarrays(
                regular_grid_coords.T, formats="f4, f4, f4")

            missing_coords = np.setdiff1d(rgc_coords, sd_coords,
                                          assume_unique=True)

            # due to float inaccuracy errors, sorted_data may not be a strict
            # subset of regular_grid_coords
            if len(missing_coords) != len(rgc_coords) - len(sd_coords):
                log.error("FIXME float inaccuracy errors")
                log.debug(f"{len(missing_coords)} {len(rgc_coords)} {len(sd_coords)}") 
                breakpoint() 
                raise Exception()

            new_data = self._gen_band_data_array(
                len(regular_grid_coords), k_coords=True, index=True)

            # add existing data to the beginning
            new_data[:len(data)]['k'] = data['k']
            new_data[:len(data)]['idx'] = data['idx']

            # add missing coordinates after
            new_data[len(data):]['k'] = missing_coords.view('3f4')

            if missing_coords_strategy == 'nan':
                new_data[len(data):]['idx'] = -1
            if missing_coords_strategy == 'backfold':
                # find exact matches
                all_k = new_data['k'].copy()
                all_k = (all_k @ self.run.primitive_lattice.reciprocal_lattice.inv_matrix + 1e-4) % 1 -1e-4 # k to reciprocal lattice vectors parallelepiped
                all_k = all_k @ self.run.primitive_lattice.reciprocal_lattice.matrix

                lattice_ijk = all_k @ lattice.inv_matrix
                lattice_ijk_int = np.rint(lattice_ijk).astype(int)
                assert np.abs(lattice_ijk_int-lattice_ijk).max() < 1e-4, 'detect_grid should have caught this'

                k_u, idx_u, inv_u = np.unique(
                    lattice_ijk_int, axis=0,
                    return_index=True, return_inverse=True)
                    
                # k_u: unique values of all_k
                # idx_u: indices of unique values of all_k
                # inv_u: indices of origin values on k_u (k_u[inv_u] == all_k)

                # assert that all missing coordinates have an exact match in
                # data.
                existing_data_indices = np.arange(len(data))
                missing_data_indices = np.setdiff1d(idx_u, existing_data_indices)
                log.debug('mdi {}', missing_data_indices)

                missing_idx = idx_u >= len(data)
                # make sure all unique k are contained in available data
                assert not any(missing_idx), f"{sum(missing_idx)} k not found" 

                new_data['idx'] = new_data['idx'][idx_u][inv_u]

            new_k = new_data['k']
            nsd_idx = np.lexsort((new_k[:, 2], new_k[:, 1], new_k[:, 0]))
            new_sorted_data = new_data[nsd_idx]
            assert np.array_equal(new_sorted_data['k'], regular_grid_coords)
            return axes, new_sorted_data.reshape(*shape)['idx']

    def get_interpolator(self, data=None, kind=None, bands=None):
        """Returns an interpolator that accepts sampling points of shape (..., 3)
        and returns energy levels of shape (..., n_e)"""

        if data is None or data == 'padded_symm':
            data = self.padded_symm_data
        elif data == 'symm':
            data = self.symm_data
        elif data in ('raw', 'data'):
            data = self._gen_band_data_array(
                len(self.data), k_coords=True, index=True)
            data['k'] = self.data['k']
            data['idx'] = np.arange(len(self.data))
            data = remove_duplicates(data)

        rgd = None
        if kind is None:
            rgd = self.reshape_gridded_data(data)
            kind = 'tri' if rgd is None else 'rect'

        if kind == 'rect':
            if rgd is None:
                rgd = self.reshape_gridded_data(data)

            log.info('Rectilinear interpolation')
            axes, data_idx = rgd
            data_e = self.data[data_idx]['e'][..., bands]
            return RegularGridInterpolator(axes, data_e)
        elif kind == 'tri':
            log.info('Triangulated interpolation')
            data_e = self.data[data['idx']]['e'][..., bands]
            return LinearNDInterpolator(data['k'], data_e)


    @cached_property
    def interpolator(self):
        """Default interpolator from Band.get_interpolator()"""
        return self.get_interpolator()

    @staticmethod
    def smooth_overlap(arr_e, e=0., scale=0.02, axis=2, weights=None):
        """
        Calculates the Gaussian overlap of the energy levels in arr_e with the Fermi level or desired energy level `e`.
        Useful for generating smooth Fermi surface pseudospectra.

        :param arr_e: array containing energy levels of shape (..., n_e)
        :param e: energy level to calculate overlap for (default: 0)
        :param scale: scale of the gaussian (default: 0.02)
        :param axis: extra axis or tuple of axes along which the energy levels are summed (default: -1)
        :param weights: weights for each energy level, should have shape compatible with e_k_3d (default: None)
        """
        arr_e[np.isnan(arr_e)] = -np.inf
        t1 = norm.pdf(arr_e, loc=e, scale=scale)
        sum_axis = (-1,) if axis is None else (axis, -1)

        if weights is not None:
            weights = np.ones_like(arr_e) * weights
        return np.average(t1, axis=sum_axis, weights=weights)

    @classmethod
    def apply_symmetry(cls, data, symm_ops, basis=None):
        """
        Applies the given symmetry operations to the k-space coordinates in
        `data` and returns a structured array with fields `k` and `idx`.

        `k` is the coordinate of the band data in k-space.
        `ìdx` is an index of `data`, where the band data equivalent to `k` can
        be found.
        """

        k_points = data['k']
        if basis is not None:
            k_points = k_points @ np.linalg.inv(basis)  # from cartesian to basis vectors
        num_k = len(k_points)

        k_idx = np.arange(num_k)
        new_data_idx = cls._gen_band_data_array(num_k*len(symm_ops),
                                                k_coords=True, index=True)
        for i, op in enumerate(symm_ops):
            new_k_points = k_points @ op.T    # .T since we are multiplying from the right
            new_data_idx['k'][num_k*i:num_k*(i+1)] = new_k_points
            new_data_idx['idx'][num_k*i:num_k*(i+1)] = k_idx

        log.debug('applied {} symm ops to {} k points', len(symm_ops), num_k)
        new_data_idx = remove_duplicates(new_data_idx)

        if basis is not None:
            new_data_idx['k'] = new_data_idx['k'] @ basis # from basis vectors to cartesian

        return new_data_idx


class Hamiltonian(FPLOFile):
    __fplo_file__ = ("+hamdata",)
    _sections = ('RTG', 'lattice_vectors', 'centering', 'fullrelativistic', 'have_spin_info', 'nwan', 'nspin',
           'wannames', 'wancenters')

    @loads(*_sections, 'data_raw')
    def load(self):
        hamdata_file = open(self.filepath, 'r')

        sections = iter(self._sections)
        s = next(sections)
        data = []
        section_data = None

        for i, line in enumerate(hamdata_file):
            if line.startswith("{}:".format(s)):
                data.append(section_data)
                try:
                    s = next(sections)
                except StopIteration:
                    s = 'spin'
                section_data = ""
            else:
                section_data += line
        data.append(section_data)

        print(list(zip(self._sections, data[1:])))

        return data[1:]

    @cached_property
    def matrix(self):
        data_raw = self.data_raw

        ftype = np.float32
        dtype = []
        dtype.append(('Tij', ftype, (3,)))
        dtype.append(('Hij', np.csingle))

        #matrix = np.zeros((nwan, nwan), dtype)


        re.compile(r"(?<=Tij, Hij:\n)^(end Tij, Hij:\n)*(?=end Tij, Hij:\n)", flags=re.MULTILINE)


        return "yolo"

    @cached_property
    def hoppings(self):
        pass