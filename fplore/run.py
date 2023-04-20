# -*- coding: utf-8 -*-

import os
from functools import cached_property

import numpy as np
from scipy.spatial.transform import Rotation
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.groups import SpaceGroup, sg_symbol_from_int_number
from pymatgen.symmetry.bandstructure import HighSymmKpath

from .logging import log
from .util import backfold_k, rot_v1_v2
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
    def spacegroup_symbol(self):
        """Returns Hermann-Mauguin symbol including setting."""
        symb = sg_symbol_from_int_number(self.spacegroup_number)
        symb = symb[:-1] if symb.endswith('H') else symb
        setting = None
        if self.spacegroup_setting:
            setting = ":" + self.spacegroup_setting
        return f"{symb}{setting or ''}"

    @property
    def spacegroup_setting(self):
        try:
            setting = self["=.in"].structure_information.spacegroup.setting
        except AttributeError:
            setting = None

        return setting if setting != "default" else None

    @property
    def spacegroup(self):
        #sg_symbol = sg_symbol_from_int_number(self.spacegroup_number)
        #setting = f":{self.spacegroup_setting}" if self.spacegroup_setting else ""
        return SpaceGroup(self.spacegroup_symbol)

    @property
    def cellrotation(self):
        """Reproduces the cell rotation matrix as present in XFPLO structure dialog."""
        try:
            cellrotation = self["=.in"].structure_information.cellrotation
        except AttributeError:
            return np.eye(3)

        if cellrotation.active is False:
            return np.eye(3)
        Rmat = np.zeros((3, 3))
        Rmat[0] = cellrotation.newx
        Rmat[2] = cellrotation.newz
        Rmat[1] = np.cross(Rmat[2], Rmat[0])
        return Rmat

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
        if self.spacegroup.crystal_system in ('trigonal', 'hexagonal') and self.spacegroup_setting != "R":
            lattice = Lattice(lattice.matrix @ Rotation.from_rotvec([0, 0, 30], degrees=True).as_matrix())
        elif self.spacegroup.crystal_system == 'trigonal' and self.spacegroup_setting == "R":
            # pymatgen will put rhombohedral c axis to 001, a, b to ???
            # but FPLO puts hexagonal c axis to 001 also in rhombohedral setting
            conventional_c = lattice.matrix.sum(axis=0)/3
            R_alignc = rot_v1_v2(v1=conventional_c, v2=np.array([0, 0, 1]))
            # a projected onto xy plane should point along x
            a = lattice.matrix[0] @ R_alignc.T
            a_proj = a - (a @ np.array([0, 0, 1])) * np.array([0, 0, 1])
            R_aligna = rot_v1_v2(v1=a_proj, v2=np.array([1, 0, 0]))
            R = R_aligna @ R_alignc
            lattice = Lattice(lattice.matrix @ R.T)
        elif self.spacegroup.crystal_system not in ('cubic', 'tetragonal', 'orthorhombic'):
            log.warning('untested lattice, crystal orientation may not be correct')

        # apply cell rotation
        lattice = Lattice(lattice.matrix @ self.cellrotation)

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
            self.spacegroup_symbol, self.lattice, elements, coords)

        return structure

    @cached_property
    def point_group_operations_frac(self):
        """Returns the point group operations in real space lattice fractional coordinates.
        Identity (E) is guaranteed to be first."""

        # note: SpacegroupAnalyzer(self.structure).get_point_group_operations(cartesian=False) gives the actual
        # symmetry and should typically give the same results, but sometimes you might intentionally be running
        # in a lower symmetry setting than the actual symmetry, so we use the declared symmetry instead.

        seen = set()
        pg_ops = [op.rotation_matrix for op in self.spacegroup.symmetry_ops
                  if not op.rotation_matrix.tobytes() in seen
                  and seen.add(op.rotation_matrix.tobytes()) is None]  # for uniqueness

        return sorted(pg_ops, key=lambda x: np.linalg.norm(x - np.eye(3)))  # ensure identity is first

    @cached_property
    def point_group_operations(self):
        """Returns the point group operations in cartesian coordinates.
        Identity (E) is guaranteed to be first."""

        pg_ops_cart = []
        for op_frac in self.point_group_operations_frac:
            op = self.lattice.matrix.T @ op_frac @ self.lattice.inv_matrix.T  # fractional -> cartesian
            op[abs(op) < 1e-12] = 0  # nearly zero
            nearly_1 = (abs(abs(op) - 1) < 1e-12)  # nearly +-1
            op[nearly_1] = np.rint(op[nearly_1])
            pg_ops_cart.append(op)
        return pg_ops_cart

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
