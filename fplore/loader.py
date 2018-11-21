# encoding: utf8
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
from six import with_metaclass
import re
import itertools
from collections import Counter, OrderedDict

import numpy as np
import progressbar
from scipy.stats.distributions import norm
from scipy.spatial import KDTree
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from cached_property import cached_property
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.groups import (SpaceGroup, PointGroup,
                                      sg_symbol_from_int_number)
from pymatgen.symmetry.bandstructure import HighSymmKpath

from . import log
from .config import FPLOConfig
from .util import cartesian_product

nbs = (0, 1, -1)
neighbours = list(itertools.product(nbs, nbs, nbs))
RegexType = type(re.compile(''))


class FPLOFileException(Exception):
    pass


class FPLOFileType(type):
    def __init__(cls, name, bases, attrs):
        def register_loader(filename):
            cls.registry['loaders'][filename] = cls

        fplo_file = getattr(cls, '__fplo_file__', None)

        if fplo_file:
            if isinstance(fplo_file, str):
                register_loader(fplo_file)

            elif isinstance(fplo_file, RegexType):
                cls.registry['loaders_re'][fplo_file] = cls
            else:
                for f in fplo_file:
                    register_loader(f)


class FPLOFile(with_metaclass(FPLOFileType, object)):
    registry = {'loaders': {}, 'loaders_re': OrderedDict()}
    is_loaded = False
    load_default = False

    @classmethod
    def get_file_class(cls, path):
        fname = os.path.basename(path)
        try:
            return cls.registry['loaders'][fname]
        except KeyError:
            for rgx, loader in cls.registry['loaders_re'].items():
                if rgx.fullmatch(fname):
                    return loader
            raise

    @classmethod
    def open(cls, path, load=False, run=None):
        if os.path.isdir(path):
            return FPLORun(path)

        FileClass = cls.get_file_class(path)
        file_obj = FileClass(path, run=run)
        if load or (load is None and cls.load_default):
            file_obj.load()

        return file_obj

    @classmethod
    def load(cls, path):
        cls.open(path, load=True)

    def __load(self):
        if self.is_loaded:
            log.notice('Reloading {}', self.filepath)

        try:
            self._load()
        except KeyError:
            raise FPLOFileException(
                "FPLO file class {} has no '_load' function".format(
                    self.__name__))

        self.is_loaded = True

    def __init__(self, filepath, run=None):
        self.load = self.__load
        self.filepath = filepath
        self.run = run


class Error(FPLOFile):
    __fplo_file__ = "+error"
    load_default = True

    def _load(self):
        self.messages = open(self.filepath, 'r').read()

        if self.messages.strip() != "":
            log.warning('+error file not empty:\n{}', self.messages)


class Run(FPLOFile):
    __fplo_file__ = "+run"
    load_default = True

    def _load(self):
        self.attrs = {}
        with open(self.filepath, 'r') as run_file:
            for line in run_file:
                key, value = line.split(':', 1)
                self.attrs[key.strip()] = value.strip()


class DOS(FPLOFile):
    __fplo_file__ = re.compile("\+dos\..+")

    def _load(self):
        dos_file = open(self.filepath, 'r')

        header = next(dos_file)
        # todo: parse header & filename

        data = []
        for line in dos_file:
            ls = line.split()
            if len(ls) == 2:
                data.append(tuple(map(float, line.split())))

        self.data = np.array(data, dtype=[
            ('e', 'f4'),
            ('dos', 'f4'),
        ])


class Dens(FPLOConfig, FPLOFile):
    __fplo_file__ = "=.dens"


class Points(FPLOFile):
    __fplo_file__ = "+points"

    def _load(self):
        points_file = open(self.filepath, 'r')

        n_points = int(next(points_file).split()[1])
        lines_per_point = 4

        self.data = []

        for lines in itertools.zip_longest(*[points_file] * lines_per_point):
            label_match = re.match("^# ' (.*) '$", lines[0])
            label = label_match.group(1)
            ik = float(lines[1].split()[0])
            self.data.append((ik, label))

        # todo: why are there 2 lines, and what's the second number?


# todo unify/subclass Band parser
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

    @staticmethod
    def _gen_band_data_array(num_k, num_e):
        return np.zeros(num_k, dtype=[
            ('ik', 'f4'),
            ('k', '3f4'),#[
                #('x', 'f4'),
                #('y', 'f4'),
                #('z', 'f4')]),
            ('e', 'f4', (num_e,)),
        ])

    def _load(self):
        band_kp_file = open(self.filepath, 'r')
        header_str = next(band_kp_file)

        log.debug(header_str)
        _0, _1, n_k, _3, n_bands, n_spinstates, _6, size2 = (
            f(x) for f, x in zip((int, float, int, int, int, int, int, int),
                                 header_str.split()[1:]))

        bar = progressbar.ProgressBar(max_value=n_k)

        # k and e appear to be 4-byte (32 bit) floats
        self.data = self._gen_band_data_array(n_k, n_bands)

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

            self.data[i]['ik'] = ik
            self.data[i]['k'] = k
            self.data[i]['e'] = e

        log.info('Band data is {} MiB in size',
                 self.data.nbytes / (1024 * 1024))

    def reshape(self, dimensions=None):
        shape = dimensions or self.shape()
        if shape == 1:
            return self.data

        if not self.is_rectangular_grid():
            raise Exception("can't handle uneven arrays yet")

        if shape == 2:
            return self.as_2d()
        if shape == 3:
            return self.as_3d()

    @cached_property
    def reshaped_data(self):
        # todo: apply symmetries to make semi-regular grid regular
        # todo: add 2d reshape ability
        # check if sample points form regular grid
        # normalise ordering
        # reshape
        # make interpolator

        xs = sorted(np.unique(self.data['k'][:, 0]))
        ys = sorted(np.unique(self.data['k'][:, 1]))
        zs = sorted(np.unique(self.data['k'][:, 2]))
        axes = xs, ys, zs
        shape = len(xs), len(ys), len(zs)

        k = self.data['k']
        sorted_data = self.data[np.lexsort((k[:, 2], k[:, 1], k[:, 0]))]
        # sorted_k = sorted(self.data['k'], key=lambda x: (x[2], x[1], x[0]))

        regular_grid_coords = cartesian_product(*axes)

        if not np.array_equal(sorted_data['k'], regular_grid_coords):
            log.debug('detected irregular k-sample grid')
            return

        log.debug('detected regular k-sample grid of shape {}', shape)

        return axes, sorted_data.reshape(*shape)

    @cached_property
    def interpolator(self):
        from scipy.interpolate import RegularGridInterpolator

        if self.reshaped_data is None:
            return

        axes, data = self.reshaped_data
        return RegularGridInterpolator(axes, data['e'])

    def to_grid(self):
        """Create a regular grid by sampling"""
        from scipy.interpolate import griddata

        # define grid.
        xi = np.linspace(np.min(self.data['k'][:, 0]), np.max(self.data['k'][:, 0]), 100)
        yi = np.linspace(np.min(self.data['k'][:, 1]), np.max(self.data['k'][:, 1]), 100)
        zi = np.linspace(np.min(self.data['k'][:, 2]), np.max(self.data['k'][:, 2]), 100)

        log.debug('xi {}..{}', xi[0], xi[-1])
        log.debug('yi {}..{}', yi[0], yi[-1])
        log.debug('zi {}..{}', zi[0], zi[-1])

        X, Y, Z = np.meshgrid(xi, yi, zi)

        # unstructure data
        k_points = self.data['k']

        # grid the data.
        rv = griddata(k_points, self.data['e'][:, 0], (X, Y, Z), method='linear')


        self.plot_k_points()
        print(rv)
        raise Exception()
        return rv

        def k_to_i(k):
            """Transform k-space coordinates to array indices"""
            kmin = np.array([kx_space[0], ky_space[0], kz_space[0]])
            kmax = np.array([kx_space[-1], ky_space[-1], kz_space[-1]])
            N = np.array((Nx, Ny, Nz))
            
            if not ((kmin <= k).all() and (k <= kmax).all()):
                raise ValueError
            
            
            t = (k-kmin) / (kmax - kmin)
            
            # handle kmin == kmax
            const = np.logical_and(kmax==kmin, kmin==k)
            t[const] = 0
            
            i = t*(N-1)

            #assert(k == k_points_3d[i])
            return i
        
        self.k_to_i = k_to_i

        self.index_z0 = np.argmin(np.abs(kz_unique))

    def bands_at_energy(self, e=0, tol=0.05):
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


class InFile(FPLOConfig, FPLOFile):
    __fplo_file__ = "=.in"


class SymFile(FPLOConfig, FPLOFile):
    __fplo_file__ = "=.sym"


def backfold_k(A, b):
    """
    Wraps an array of k-points b (shape (n_points, 3)) back to the first
    Brillouin zone given a reciprocal lattice matrix A.

    Note: Assumes that the lattice vectors contained in A correspond to the shortest
    lattice vectors, i.e. that pairwise combinations of reciprocal lattice
    vectors in A and their negatives cover all the nearest neighbours of the
    BZ.
    """

    # get adjacent BZ cell's Gamma point locations
    neighbours_k = np.dot(np.array(neighbours), A)

    from scipy.spatial.distance import cdist

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

        # assign backfolded coordinates to output array
        b[idx_requires_backfolding] = backfolded

        # only those coordinates which were changed in this round need to be
        # backfolded again
        idx_requires_backfolding = idx_requires_backfolding[idx_backfolded]


def remove_duplicates(data):
    """Remove non-unique k-points from data"""
    # todo: float tolerance

    k_points = set()
    duplicates = []

    for i, d in enumerate(data):
        kp = tuple(d['k'])
        if kp in k_points:
            duplicates.append(i)
        else:
            k_points.add(kp)

    log.debug('deleting {} duplicates from band data', len(duplicates))
    return np.delete(data, duplicates, axis=0)


def apply_symmetry(data, symm_ops):
    k_points = data['k']

    num_k, num_e = data['e'].shape
    new_data = Band._gen_band_data_array(num_k*len(symm_ops), num_e)
    for i, op in enumerate(symm_ops):
        rot = op.rotation_matrix
        new_k_points = np.dot(rot, k_points.T).T
        new_data['k'][num_k*i:num_k*(i+1)] = new_k_points
        new_data['e'][num_k*i:num_k*(i+1)] = data['e']

    log.debug('applied {} symm ops to {} k points', len(symm_ops), num_k)
    return remove_duplicates(new_data)


class FPLORun(object):
    def __init__(self, directory):
        log.debug("Initialising FPLO run in directory {}", directory)

        self.directory = directory
        self.files = {}

        # print available files for debug purposes
        fnames = [f for f in os.listdir(directory)
                  if os.path.isfile(os.path.join(directory, f))]
        loaded = set()
        for fname in fnames:
            try:
                self.files[fname] = FPLOFile.open(
                    os.path.join(directory, fname), run=self)
            except KeyError as e:
                pass
            else:
                if self.files[fname].load_default:
                    self.files[fname].load()
                    loaded.add(fname)

        log.info("Loaded files: {}", ", ".join(sorted(loaded)))
        log.info("Loadable files: {}", ", ".join(sorted(
            self.files.keys() - loaded)))
        log.debug("Not loadable: {}", ", ".join(sorted(
            set(fnames) - self.files.keys())))

    def __getitem__(self, item):
        f = self.files[item]
        if not f.is_loaded:
            log.debug('Loading {} due to getitem access via FPLORun', item)
            f.load()
        return f

    @property
    def attrs(self):
        return self["+run"].attrs

    @property
    def spacegroup_number(self):
        return self["=.in"].structure_information.spacegroup.number

    @property
    def spacegroup(self):
        sg_symbol = sg_symbol_from_int_number(self.spacegroup_number)
        return SpaceGroup(sg_symbol)

    @property
    def lattice(self):
        si = self["=.in"].structure_information

        # todo: convert non-angstrom units
        assert si.lengthunit.type == 2

        lattice = Lattice.from_lengths_and_angles(
            abc=si.lattice_constants,
            ang=si.axis_angles)

        return lattice

    @property
    def structure(self):
        si = self["=.in"].structure_information

        elements = []
        coords = []
        for wp in si.wyckoff_positions:
            elements.append(wp.element)
            coords.append([float(x) for x in wp.tau])

        structure = Structure.from_spacegroup(
            self.spacegroup_number, self.lattice, elements, coords)

        return structure

    @property
    def primitive_structure(self):
        return self.structure.get_primitive_structure()

    @property
    def primitive_lattice(self):
        return self.primitive_structure.lattice

    @property
    def brillouin_zone(self, primitive=True):
        if primitive:
            return self.primitive_lattice.get_brillouin_zone()
        return self.lattice.get_brillouin_zone()

    @cached_property
    def band_data(self):
        """Returns the band data folded back to the first BZ"""
        try:
            band = self['+band']
        except KeyError:
            band = self['+band_kp']

        # convert fractional coordinates to k-space coordinates
        points = self.frac_to_k(band.data['k'])

        # wrap points to primitive unit cell BZ
        points = backfold_k(self.primitive_lattice.reciprocal_lattice.matrix,
                            points)

        data = band.data.copy()
        data['k'] = points

        return data

    @property
    def symm_band_data(self):
        """Returns the band data folded back to the first BZ and applies
        symmetry operations"""
        pg = PointGroup(self.spacegroup.point_group)

        # apply symmetry operations from point group
        data = np.array(apply_symmetry(self.band_data, pg.symmetry_ops))

        return data

    # todo: k-coordinate array class which automatically wraps back to first bz
    #       and irreducible wedge

    @property
    def neigh_band_data(self):
        bd = self.symm_band_data
        k_points = bd['k']
        num_k, num_e = bd['e'].shape
        new_data = Band._gen_band_data_array(num_k * len(neighbours), num_e)
        for i, nb in enumerate(neighbours):
            offset = np.dot(self.primitive_lattice.reciprocal_lattice.matrix,
                            nb).T
            new_data['k'][num_k * i:num_k * (i + 1)] = k_points + offset
            new_data['e'][num_k * i:num_k * (i + 1)] = bd['e']

        return new_data

    @property
    def band_data_dimension(self):
        # todo: not reliable, needs lots of memory for large data sets
        """Attempts to determine the dimensionality of band information, i.e.
        whether the k-points lie on a path, a plane, or span a volume."""

        k_points = self.symm_band_data['k']
        if len(k_points) < 1000:
            neigh_k_points = self.neigh_band_data['k']
        else:
            neigh_k_points = k_points

        # fit function for number of neighbours within r
        def nn(r, a, e):
            return a * r**e

        kdt = KDTree(neigh_k_points)

        num_p_per_sample = min(len(neigh_k_points)//2, (1+2*2)**3)
        num_samples = min(int(np.sqrt(len(k_points))), 100)

        dimensionality = []
        i_ps = np.random.choice(range(len(k_points)), num_samples,
                               replace=False)

        for i_p in i_ps:
            p = k_points[i_p]
            _dists, _ = kdt.query(p, num_p_per_sample)
            _dists = _dists.round(6)  # round to 6 decimal places

            cum_num_p = {}
            n = 0
            c = Counter(_dists)
            for dist in sorted(c.keys()):
                if dist == 0:
                    continue
                n += c[dist]
                cum_num_p[dist] = n

            dists, num_p = zip(*sorted(cum_num_p.items()))
            popt, pcov = curve_fit(nn, dists, num_p,
                                   bounds=(0, np.inf))
            perr = np.sqrt(np.diag(pcov))
            log.debug('e {}+-{} p {}', popt[1], perr[1], p)
            dimensionality.append(popt[1])

        avg = np.average(dimensionality)
        std = np.std(dimensionality)

        spread = (avg - std, avg + std)
        log.debug("dimension spread {}", spread)

        if abs(int(spread[0]) - int(spread[1])) > 1:
            log.error('Dimensionality unreliable.')
            raise Exception('Dimensionality unreliable.')

        if int(spread[0]) == int(spread[1]):  #
            log.warn('Dimension uncertainty does not contain integer dimension.')

        log.debug('avg {}', avg)
        return int(avg + std)

    def band_data_sample(self, sample_points):
        data = self.symm_band_data

        # todo: detect if regular grid and use interpn instead of griddata
        # plan: get extent of k points
        #       generate regular grid
        #       wrap k points
        #       if wrapped == data['k'] then it is a regular grid

        log.debug('samping band data at {} points', len(sample_points))
        return griddata(data['k'], data['e'], sample_points, method='nearest')

    def plot_structure(self):
        raise NotImplementedError

    def plot_bz(self, ax, vectors=True, k_points=False, use_symmetry=False,
                high_symm_points=True):
        # todo: move this function somewhere more appropriate?
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib import colors as mcolors

        def cc(arg):
            return mcolors.to_rgba(arg, alpha=0.1)

        if k_points:
            if use_symmetry:
                points = self.symm_band_data['k']
            else:
                points = self.band_data['k']

            ax.plot(*points.T, '.',
                    label='sample k-point', ms=1)

        if vectors:
            from .plot import Arrow3D
            for vec in self.primitive_lattice.reciprocal_lattice.matrix:
                ax.add_artist(Arrow3D(*zip((0, 0, 0), vec), mutation_scale=20,
                                      lw=3, arrowstyle="-|>", color="r"))

        ax.add_collection3d(
            Poly3DCollection(self.brillouin_zone, facecolors=cc('k'),
                             edgecolors='k'))

        if high_symm_points:
            points = self.high_symm_kpoints
            ax.plot(*zip(*points.values()), 'o', label='high symmetry point', color='k', ms='1')

            for kpath in self.high_symm_kpaths:
                path = [points[lbl] for lbl in kpath]
                ax.plot(*zip(*path), '-', color='k', alpha=0.5)

            for label, coord in points.items():
                ax.text(*coord, '${}$'.format(label), color='k')

        ax.set_xlabel('$k_x$')
        ax.set_ylabel('$k_y$')
        ax.set_zlabel('$k_z$')
        ax.legend()

    @cached_property
    def high_symm_kpaths(self):
        return HighSymmKpath(self.primitive_structure).kpath['path']

    @cached_property
    def high_symm_kpoints_fractional(self):
        return HighSymmKpath(self.primitive_structure).kpath['kpoints']

    @cached_property
    def high_symm_kpoints(self):
        points = self.high_symm_kpoints_fractional
        for label, coord in points.items():
            points[label] = np.dot(
                coord, self.primitive_lattice.reciprocal_lattice.matrix)
        return points

    def frac_to_k(self, fractional_coords):
        """
        Transforms fractional lattice coordinates to k-space coordinates.

        :param fractional_coords: Nx3
        :return: k_points: Nx3
        """

        # coordinates are in terms of conventional unit cell BZ, not primitive
        return np.dot(fractional_coords,
                      self.lattice.reciprocal_lattice.matrix)

    # todo: move to util
    def linspace_ng(self, start, *stops, num=50,
                    fractional_coordinates=True):
        """
        Return evenly spaced coordinates between n arbitrary 3d points

        A single stop will return a line between start and stop, two stops make
        a plane segment, three stops will span a parallelepiped.

        """
        if fractional_coordinates:
            hskp = self.high_symm_kpoints_fractional
        else:
            hskp = self.high_symm_kpoints

        if isinstance(start, str):
            start = hskp[start]

        for i, stop in enumerate(stops):
            if isinstance(stop, str):
                stops[i] = hskp[stop]

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
