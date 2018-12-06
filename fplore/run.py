# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os

import numpy as np
from cached_property import cached_property
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.groups import SpaceGroup, sg_symbol_from_int_number
from pymatgen.symmetry.bandstructure import HighSymmKpath

from .logging import log
from .util import backfold_k
from .files.base import FPLOFile


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
    def band(self):
        """Returns the band data file"""
        try:
            band = self['+band']
        except KeyError:
            band = self['+band_kp']

        return band

    # todo: k-coordinate array class which automatically wraps back to first bz
    #       and irreducible wedge

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

    def backfold_k(self, points):
        return backfold_k(
            self.primitive_lattice.reciprocal_lattice.matrix, points)

    def frac_to_k(self, fractional_coords):
        """
        Transforms fractional lattice coordinates to k-space coordinates.

        :param fractional_coords: Nx3
        :return: k_points: Nx3
        """

        # coordinates are in terms of conventional unit cell BZ, not primitive
        return np.dot(fractional_coords,
                      self.lattice.reciprocal_lattice.matrix)

    def k_to_frac(self, k_coords):
        return np.dot(k_coords,
                      self.lattice.reciprocal_lattice.inv_matrix)
