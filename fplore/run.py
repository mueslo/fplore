# -*- coding: utf-8 -*-

import os
from functools import cached_property

import numpy as np
from scipy.spatial.transform import Rotation
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.groups import SpaceGroup, sg_symbol_from_int_number
from pymatgen.symmetry.bandstructure import HighSymmKpath

from .logging import log
from .util import backfold_k
from .files.base import FPLOFile
from .files import DOS


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
            except KeyError:
                pass
            else:
                if self.files[fname].load_default:
                    self.files[fname].load()
                    loaded.add(fname)

        log.info("Loaded files: {}", ", ".join(sorted(loaded)))
        log.info("Loadable files: {}", ", ".join(sorted(
            set(self.files.keys()) - loaded)))
        log.debug("Not loadable: {}", ", ".join(sorted(
            set(fnames) - set(self.files.keys()))))

    def __getitem__(self, item):
        f = self.files[item]
        if not f.is_loaded:
            log.debug('Loading {} due to getitem access via FPLORun', item)
            f.load()
        return f

    def __repr__(self):
        return "{}('{}')".format(type(self).__name__, self.directory)

    @property
    def attrs(self):
        return self["+run"].attrs

    @property
    def spacegroup_number(self):
        return int(self["=.in"].structure_information.spacegroup.number)

    @property
    def spacegroup(self):
        sg_symbol = sg_symbol_from_int_number(self.spacegroup_number)
        return SpaceGroup(sg_symbol)

    @property
    def lattice(self):
        # lattice matrix: basis vectors are rows
        si = self["=.in"].structure_information

        # todo: convert non-angstrom units
        assert si.lengthunit.type == 2

        lattice = Lattice.from_parameters(
            *si.lattice_constants, 
            *si.axis_angles)

        # translate to FPLO convention
        # see also: https://www.listserv.dfn.de/sympa/arc/fplo-users/2020-01/msg00002.html
        if self.spacegroup.crystal_system in ('trigonal', 'hexagonal'):
            lattice = Lattice(lattice.matrix @ Rotation.from_rotvec([0, 0, 30], degrees=True).as_matrix())
        elif self.spacegroup.crystal_system not in ('cubic', 'tetragonal', 'orthorhombic'):
            log.warning('untested lattice, crystal orientation may not be correct')

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
    def brillouin_zone(self):
        return self.primitive_lattice.get_brillouin_zone()

    @cached_property
    def band(self):
        """Returns the band data file"""
        try:
            return self['+band']
        except KeyError:
            return self['+band_kp']

    # todo: k-coordinate array class which automatically wraps back to first bz
    #       and irreducible wedge

    @cached_property
    def band_weights(self):
        try:
            return self['+bweights']
        except KeyError:
            raise AttributeError

    def dos(self, **kwargs):
        dos_files = filter(lambda x: isinstance(x, DOS), self.files)
        print(dos_files)
        raise NotImplementedError

    @cached_property
    def high_symm_kpaths(self):
        return HighSymmKpath(self.primitive_structure).kpath['path']

    @cached_property
    def high_symm_kpoints_fractional(self):
        return HighSymmKpath(self.primitive_structure).kpath['kpoints']

    @cached_property
    def high_symm_kpoints(self):
        points_frac = self.high_symm_kpoints_fractional
        points_cart = {}
        for label, coord in points_frac.items():
            points_cart[label] = coord @ self.primitive_lattice.reciprocal_lattice.matrix
        return points_cart

    def backfold_k(self, points):
        return backfold_k(
            self.primitive_lattice.reciprocal_lattice, points)

    def fplo_to_k(self, fplo_coords):
        """
        Transforms fplo fractional lattice coordinates (units 2pi/a) to k-space coordinates.

        :param fractional_coords: Nx3
        :return: k_points: Nx3
        """

        return 2*np.pi/self.lattice.a * fplo_coords

    def _frac_to_k(self, fractional_coords):
        """
        Transforms fractional reciprocal lattice coordinates to k-space coordinates.

        :param fractional_coords: Nx3
        :return: k_points: Nx3
        """

        # coordinates are in terms of conventional unit cell BZ, not primitive
        return fractional_coords @ self.lattice.reciprocal_lattice.matrix

    def _k_to_frac(self, k_coords):
        return k_coords @ self.lattice.reciprocal_lattice.inv_matrix
