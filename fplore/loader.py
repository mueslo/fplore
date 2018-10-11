# encoding: utf8
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
from six import with_metaclass
import re
import itertools
from collections import Counter

import numpy as np
import progressbar
from scipy.stats.distributions import norm
from scipy.spatial import KDTree
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from scipy.spatial import ConvexHull
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.groups import (SpaceGroup, PointGroup,
                                      sg_symbol_from_int_number)

from . import log
from .config import FPLOConfig

nbs = (0, 1, -1)
neighbours = list(itertools.product(nbs, nbs, nbs))


class FPLOFileType(type):
    def __init__(cls, name, bases, attrs):
        def register_loader(filename):
            cls.registry['loaders'][filename] = cls

        # todo: regex filenames, e.g. "+dos.*"
        fplo_file = getattr(cls, '__fplo_file__', None)

        if fplo_file:
            if isinstance(fplo_file, str):
                register_loader(fplo_file)
            else:
                for f in fplo_file:
                    register_loader(f)


class FPLOFile(with_metaclass(FPLOFileType, object)):
    registry = {'loaders': {}}

    @classmethod
    def get_loader(cls, path):
        fname = os.path.basename(path)
        return cls.registry['loaders'][fname]

    @classmethod
    def load(cls, path):
        if os.path.isdir(path):
            return FPLORun(path)

        loader = cls.get_loader(path)
        return loader(path)


class Error(FPLOFile):
    __fplo_file__ = "+error"

    def __init__(self, filepath):
        self.messages = open(filepath, 'r').read()

        if self.messages.strip() != "":
            log.warning('+error file not empty:\n{}', self.messages)


class Run(FPLOFile):
    __fplo_file__ = ("+run",)
    def __init__(self, filepath):
        self.attrs = {}
        with open(filepath, 'r') as run_file:
            for line in run_file:
                key, value = line.split(':', 1)
                self.attrs[key.strip()] = value.strip()


class DOS(FPLOFile):
    __fplo_file__ = ("dos\..*")  # todo regex
    def __init__(self, filepath):
        raise NotImplementedError


class Dens(FPLOConfig, FPLOFile):
    __fplo_file__ = "=.dens"


class Points(FPLOFile):
    __fplo_file__ = "+points"
    def __init__(self, filepath):
        points_file = open(filepath, 'r')

        n_points = int(next(points_file).split()[1])
        lines_per_point = 4

        self.data = []

        for lines in itertools.zip_longest(*[points_file] * lines_per_point):
            label_match = re.match("^# ' (.*) '$", lines[0])
            label = label_match.group(1)
            ik = float(lines[1].split()[0])
            self.data.append((ik, label))

        # todo: why are there 2 lines, and what's the second number?

class BandWeights(FPLOFile):
    __fplo_file__ = ("+bweights", "+bweights_kp")

    def __init__(self, filepath):
        weights_file = open(filepath, 'r')
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

        self.raw_data = np.zeros(n_k, dtype=[
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

            i_k = data[0]

            self.raw_data[i]['ik'] = i_k
            self.raw_data[i]['e'] = e
            self.raw_data[i]['c'] = weights

        log.info('Band weight data is {} MiB in size',
                   self.raw_data.nbytes / (1024 * 1024))
        self.data = self.raw_data


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

    def __init__(self, filepath):
        band_kp_file = open(filepath, 'r')
        header_str = next(band_kp_file)
        
        log.debug(header_str)
        _0, _1, n_k, _3, n_bands, n_spinstates, _6, size2 = (f(x) for f,x in zip((int, float, int, int, int, int, int, int), header_str.split()[1:]))

        bar = progressbar.ProgressBar(max_value=n_k)

        # k and e appear to be 4-byte (32 bit) floats
        self.raw_data = self._gen_band_data_array(n_k, n_bands)

        for i, lines in bar(enumerate(
                itertools.zip_longest(*[band_kp_file]*(1 + n_spinstates)))):
            # read two lines at once for n_spinstates=1, three for n_s=2

            # first line
            k = tuple(map(float, lines[0].split()[1:]))

            # second (+ third) line
            ik = float(lines[1].split()[0])
            e = [list(map(float, line_e.split()[1:])) for line_e in lines[1:]]

            # todo: don't ignore magnetic split
            e = e[0]  # ignore magnetic split

            self.raw_data[i]['ik'] = ik
            self.raw_data[i]['k'] = k
            self.raw_data[i]['e'] = e

        log.info('Band data is {} MiB in size',
                   self.raw_data.nbytes / (1024 * 1024))
        self.data = self.raw_data

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


    def to_grid(self):
        from scipy.interpolate import griddata

        # define grid.
        xi = np.linspace(np.min(self.data['k']['x']), np.max(self.data['k']['x']), 100)
        yi = np.linspace(np.min(self.data['k']['y']), np.max(self.data['k']['y']), 100)
        zi = np.linspace(np.min(self.data['k']['z']), np.max(self.data['k']['z']), 100)

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
        idx_band = np.any(np.abs(self.e - e) < tol, axis=0)
        return np.where(idx_band)[0] #indices of levels that come close to E_f

    @staticmethod
    def smooth_overlap(e_k_3d, e=0., scale=0.02):
        e_k_3d[np.isnan(e_k_3d)] = -np.inf
        t1 = norm.pdf(e_k_3d, loc=e, scale=scale)
        # todo interpolate axis 2

        return np.sum(t1, axis=(2, 3))


class InFile(FPLOConfig, FPLOFile):
    __fplo_file__ = "=.in"


class SymFile(FPLOConfig, FPLOFile):
    __fplo_file__ = "=.sym"


def wrap_k(A, b):
    """
    Wraps an array of k-points b (shape (n_points, 3)) back to the first
    Brillouin zone given a reciprocal lattice matrix A
    """

    n = ((np.linalg.solve(A, b.T) + 0.5) % 1) - 0.5
    return np.dot(A, n).T


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

        self.directory = directory
        self.files = {}

        # print available files for debug purposes
        fnames = [f for f in os.listdir(directory)
                  if os.path.isfile(os.path.join(directory, f))]
        loadable = set(fnames)
        for fname in fnames:
            try:
                loader = FPLOFile.get_loader(os.path.join(directory, fname))
            except KeyError as e:
                loadable.remove(fname)
            else:
                # load some files by default
                # todo: move to metaclass
                if fname.startswith("=.") or fname in ('+error', '+run'):
                    self.files[fname] = FPLOFile.load(
                        os.path.join(self.directory, fname))


        log.info("Loaded files: {}", self.files.keys())
        log.info("Loadable files: {}", loadable - set(self.files.keys()))
        log.debug("Not loadable: {}", set(fnames) - loadable)

    def __getitem__(self, item):
        try:
            return self.files[item]
        except KeyError:
            self.files[item] = FPLOFile.load(
                os.path.join(self.directory, item))
            return self.files[item]
            #raise KeyError("No loader defined for '{}'.".format(item))

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

    @property
    def band_data(self):
        """Returns the band data folded back to the first BZ"""
        band = self.files.get('+band') or self['+band_kp']

        # coordinates are in terms of conventional unit cell BZ
        points = np.dot(self.lattice.reciprocal_lattice.matrix,
                        band.data['k'].T).T

        # wrap points to primitive unit cell BZ
        points = wrap_k(self.primitive_lattice.reciprocal_lattice.matrix,
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

    def plot_bz(self, vectors=True, k_points=True, use_symmetry=False,
                high_symm_points=True):
        # todo: move this function somewhere more appropriate?
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib import colors as mcolors

        def cc(arg):
            return mcolors.to_rgba(arg, alpha=0.1)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        if k_points:
            if use_symmetry:
                points = self.symm_band_data['k']
            else:
                points = self.band_data['k']

            plt.plot(points[:, 0], points[:, 1], points[:, 2], '.',
                     label='k-point', ms=1)

        if vectors:
            from .plot import Arrow3D
            for vec in self.primitive_lattice.reciprocal_lattice.matrix:
                ax.add_artist(Arrow3D(*zip((0, 0, 0), vec), mutation_scale=20,
                                      lw=3, arrowstyle="-|>", color="r"))

        ax.add_collection3d(
            Poly3DCollection(self.brillouin_zone, facecolors=cc('k'),
                             edgecolors='k'))

        if high_symm_points:
            # todo add pymatgen.symmetry.bandstructure.HighSymmKpath
            pass

        ax.set_xlabel('$k_x$')
        ax.set_ylabel('$k_y$')
        ax.set_zlabel('$k_z$')

        plt.legend()

        plt.show()
