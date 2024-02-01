# -*- coding: utf-8 -*-

import re
from io import StringIO
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
                    find_lattice, backfold_k_parallelepiped)


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

    def bands_within(self, e_lower, e_upper=None):
        """Returns the indices of bands which are within a certain energy range"""
        if e_upper is None:
            e_upper = -e_lower
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
        # _4: number of bands
        # _5: number of spin states
        # _6: index of first band
        # _7: index of last band

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

        # convert FPLO reciprocal coordinates to k-space coordinates
        k = self.run.fplo_to_k(self._data['frac'])

        view_type = np.dtype([('k', k.dtype, (3,))])
        k_structured = k.view(view_type)[:, 0]

        return merge_arrays([self._data, k_structured], flatten=True)

    @cached_property
    def symm_data(self):
        """Returns the band data folded back to the first BZ and applies
        symmetry operations. Returns an index array to reduce memory usage."""

        # the band structure has the Laue symmetry of the crystal (Neumann's principle)
        # (and in case of degenerate spins , then also inversion symmetry, due to time reversal)
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
        ksamp_lattice = find_lattice(k)  # attempt to find a regular lattice spanning the k-points
        extra_k = fill_bz(k, self.run.primitive_lattice.reciprocal_lattice, ksamp_lattice=ksamp_lattice, pad=True)
        #extra_k, extra_ijk = pad_regular_sampling_lattice(k_fill, ksamp_lattice=ksamp_lattice)

        # now we need to determine which of the original k-points filling the BZ zone
        # these extra k-points correspond to

        #k_frac = (k @ self.run.primitive_lattice.reciprocal_lattice.inv_matrix + 1e-4) % 1 - 1e-4

        # k to reciprocal lattice vectors parallelepiped
        #extra_k_frac = (extra_k @ self.run.primitive_lattice.reciprocal_lattice.inv_matrix + 1e-4) % 1 - 1e-4
        # 1e-4 for consistency even with float inaccuracies around edges

        #lattice_ijk = extra_k_frac @ self.run.primitive_lattice.reciprocal_lattice.matrix @ ksamp_lattice.inv_matrix
        lattice_ijk = backfold_k_parallelepiped(self.run.primitive_lattice.reciprocal_lattice, extra_k) @ ksamp_lattice.inv_matrix
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
        # k to reciprocal lattice vectors parallelepiped

        # lattice_ijk maps k sample ijk values to unique index
        lattice_ijk = backfold_k_parallelepiped(self.run.primitive_lattice.reciprocal_lattice, data['k']) @ ksamp_lattice.inv_matrix
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
            log.notice('Non-orthogonal grid detected, reshape_gridded_data will return `None`')
            return None

        if np.logical_not(np.isclose(lattice.matrix, 0)).sum() > 3:
            log.debug(lattice)
            log.warning('Rotated orthogonal grid not yet implemented')
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
                all_k = backfold_k_parallelepiped(
                    self.run.primitive_lattice.reciprocal_lattice, new_data['k'])

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
        and returns energy levels of shape (..., n_e).

        If the sampling points span a rectilinear grid in cartesian coords, a
        rectilinear interpolator is returned. Otherwise, a triangulated
        interpolator is returned.
        """

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

        `basis` is a 3x3 matrix that is used to transform the k-space to a
        different basis in which the symmetry operations are applied.
        TODO: apply to operators instead, since typically they are fewer
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
            new_data_idx['k'] = new_data_idx['k'] @ basis  # from basis back to cartesian

        return new_data_idx


class Hamiltonian(FPLOFile):
    __fplo_file__ = ("+hamdata",)
    _sections = ('RTG', 'lattice_vectors', 'centering', 'fullrelativistic', 'have_spin_info', 'nwan', 'nspin',
           'wannames', 'wancenters', 'symmetry')

    @loads(*(s+'_raw' for s in _sections), 'data_raw')
    def load(self):
        with open(self.filepath, 'r') as hamdata_file:
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

        return data[1:]

    @cached_property
    def RTG(self):
        return self.RTG_raw.strip().lower() == 't'

    @cached_property
    def fullrelativistic(self):
        return self.fullrelativistic_raw.strip().lower() == 't'

    @cached_property
    def have_spin_info(self):
        return self.have_spin_info_raw.strip().lower() == 't'

    @cached_property
    def nspin(self):
        return int(self.nspin_raw)

    @cached_property
    def nwan(self):
        return int(self.nwan_raw)

    @cached_property
    def wannames(self):
        return self.wannames_raw.strip().split('\n')

    @cached_property
    def lattice_vectors(self):
        return np.genfromtxt(StringIO(self.lattice_vectors_raw), dtype=np.float64)

    @cached_property
    def centering(self):
        return np.genfromtxt(StringIO(self.centering_raw), dtype=np.float64)

    @cached_property
    def wancenters(self):
        return np.genfromtxt(StringIO(self.wancenters_raw), dtype=np.float64)

    @cached_property
    def data(self):
        blockre = re.compile(r"(?<=Tij, Hij:\n) *(?P<i>[0-9]+) +(?P<j>[0-9]+) *\n"
                             r"(?P<TH>[0-9 E+-\.\n]*)(?=end Tij, Hij:\n)", flags=re.MULTILINE)
        hop = {}
        for i, j, TH in blockre.findall(self.data_raw):
            i, j = int(i)-1, int(j)-1  # convert 1-based to native 0-based indexing
            TH = np.genfromtxt(StringIO(TH), dtype=np.float64)
            TH = TH.view([('T', np.float64, (3,)), ('H', np.cdouble)])
            if len(TH):
                TH = TH[:, 0]
            hop[(i, j)] = TH  # if i, j repeat, second one should be second spin sort
        if self.nspin != 1:
            log.warning("nspin != 1, but only last spin channel is read")
        return hop

    def cluster_matrix(self, loc_center=(0,0,0), radius=4.0):
        """Return the Hamiltonian matrix of just the cluster (within `radius`)
        around `loc_center` including all intra-cluster hopping.

        :param loc_center: location of the center of the cluster (element of self.wancenters)
        :param radius: radius of the cluster in Bohr radii (default 4.0)
        :return cluster_matrix: Hamiltonian matrix of hopping within the cluster
            blocks sorted by cluster_loc key order, intra-block sorted by cluster_idx key order
        :return cluster_loc: cluster definition
            dictionary mapping cluster locations to corresponding Wannier site index
        :return cluster_idx: Wannier site definition
            dictionary mapping Wannier site index to contained Wannier orbital indices (self.wannames/self.wancenters)
        :return uidx: inverse of cluster_idx
            array mapping Wannier orbital indices (self.wannames/self.wancenters) to cluster_idx keys

        Note: written assuming atomically centered Wannier functions. No guarantees what the output is otherwise.
        """
        ATOL = 1e-2  # float threshold below which we consider a translation vector square sum to be zero
        #NOTE: units of length appear to always be in Bohr radii in hamdata

        # find all unique locations, their indices, and their counts, self.wancenters should have no float error(!)
        # essentially a unique ID of translationally equivalent sites
        # locs: loc -> uidx
        # uidx: idx -> uidx
        locs, uidx, count = np.unique(self.wancenters, axis=0, return_inverse=True, return_counts=True)
        idx_d = {i: np.argwhere(uidx == i)[:, 0] for i in range(len(locs))}  # uidx -> idx
        uidx_center = np.nonzero((loc_center == locs).all(axis=1))[0][0]  # find index of loc_center in locs

        # generate cluster, i.e. find all ligands within radius of loc_center
        # a cluster doesn't care about translational equivalence, so we use real locations instead of uidx
        hashcoord = lambda loc: tuple(loc)  # hashable vector, locs should have no float error! no arithmetic on loc!
        cluster_loc = {hashcoord(loc_center): uidx_center}  # add central ion to cluster; loc -> uidx
        for ic in idx_d[uidx_center]:  # loop over all wannier orbitals of central ion
            for il, uidxl in enumerate(uidx):  # loop over all wannier orbitals of candidate ligands (*including* central ion for e.g. unary compounds)
                T = self.data[ic, il]['T']
                dists2 = (T ** 2).sum(axis=1)
                hops = T[np.logical_and(dists2 < radius ** 2, dists2 > ATOL)]  # filter out hops outside radius & on-site
                for hop in hops:
                    loc = locs[uidx_center] + hop
                    cluster_loc[hashcoord(loc)] = uidxl

        # get number of blocks (== number of sites)
        nblocks = len(cluster_loc)
        #blocksizes = [count[ui] for ui in cluster_loc.values()]

        # generate array of locations in cluster from cluster_loc dict
        arr_cluster_loc = np.array(list(cluster_loc.keys()))
        # get all translation vectors between cluster sites, i.e. blocks
        Tblock = arr_cluster_loc[np.newaxis] - arr_cluster_loc[:, np.newaxis]

        # get all hoppings between cluster sites, i.e. blocks
        Hblocks = {}

        # loop over all blocks
        for ib, (loci, uidxi) in enumerate(cluster_loc.items()):
            for jb, (locj, uidxj) in enumerate(list(cluster_loc.items())[ib:], ib):
                # T = locj - loci
                Hblocks[ib, jb] = Hblock = np.zeros((count[uidxi], count[uidxj]), dtype=np.cdouble)

                # loop over all hoppings between sites in block ib and jb
                for i, idxi in enumerate(idx_d[uidxi]):
                    for j, idxj in enumerate(idx_d[uidxj]):
                        # would be nicer to have an indexed version of translation vectors, but this is fine for now
                        entry = np.nonzero(((self.data[idxi, idxj]['T'] - Tblock[ib, jb]) ** 2).sum(axis=1) < ATOL)[0]

                        if entry.size:
                            assert entry.size == 1  # there should only be one matching translation vector
                            Hblock[i, j] = self.data[idxi, idxj]['H'][entry[0]]
                Hblocks[jb, ib] = Hblock.conj().T

        Hcluster = np.block([
            [Hblocks[i, j] for j in range(nblocks)] for i in range(nblocks)])

        return Hcluster, cluster_loc, idx_d, uidx

    @staticmethod
    def cluster_diag(Hcluster, n_center, transform_center=None):
        """
        Return cluster Hamiltonian in symmetry adapted basis, i.e. transforming the basis to make the
        interaction between the central ion and the ligands as diagonal as possible.

        :param Hcluster: cluster Hamiltonian matrix
        :param n_center: number of orbitals on central ion (has to be first in cluster)
        :param transform_center: if True, allow also for a unitary transformation of the cluster center site
            True or 'interaction': SVD decomposition, interaction diagonal
            False: polar decomposition, no transformation of center, interaction matrix at least symmetric
            'local': polar decomposition plus transformation of center site to diagonalize local Hamiltonian
        :return Hcluster: cluster Hamiltonian matrix in symmetry adapted basis
        :return U: unitary transformation matrix
        """

        from scipy import linalg

        Up, Sigma, Ud_H = linalg.svd(Hcluster[n_center:, :n_center])
        #Sigma = linalg.diagsvd(Sigma, len(Up), len(Um_T))

        n_ligands = len(Hcluster) - n_center
        assert n_ligands >= n_center
        if transform_center == 'local':  # diag(Ud, Up).diag(Ud^H, Ud^H, 1).diag(ev, ev, 1)
            ev = linalg.eigh(Hcluster[:n_center, :n_center])[1]
            U = linalg.block_diag(
                ev, (Up @ linalg.block_diag(Ud_H @ ev, np.eye(n_ligands - n_center))))
        elif transform_center:
            U = linalg.block_diag(Ud_H.T.conj(), Up)
        else:  # diag(Ud, Up).diag(Ud^H, Ud^H, 1)
            U = linalg.block_diag(
                np.eye(n_center), (Up @ linalg.block_diag(Ud_H, np.eye(n_ligands - n_center))))

        return U.T.conj() @ Hcluster @ U, U
