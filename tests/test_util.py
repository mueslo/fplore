import pytest_cases
import math
import numpy as np
from fplore.util import wigner_seitz_neighbours, backfold_k
from pymatgen.core import Lattice


def fibonacci_sphere(samples=1000):
    # via https://stackoverflow.com/a/26127012/925519
    points = np.arange(1000)
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians
    z = np.linspace(1, -1, samples)  # z goes from 1 to -1
    radius = np.sqrt(1 - z**2)  # radius at z
    theta = phi * points  # golden angle increment
    x = np.cos(theta) * radius
    y = np.sin(theta) * radius
    return np.vstack([x, y, z]).T


@pytest_cases.fixture
def directions():
    return fibonacci_sphere(1000)


@pytest_cases.fixture
@pytest_cases.parametrize(
    "lattice_type,params",
    [('cubic', {'a': 0.8612}),
     ('rhombohedral', {'a': 0.8612, 'alpha': 72.4}),
     ('from_parameters', {'a': 0.8612, 'b': 0.325, 'c': 0.5583, 'alpha': 42.6, 'beta': 88.2, 'gamma': 63.3})])
def reciprocal_lattice(lattice_type, params):
    # todo randomly rotated lattices?
    return getattr(Lattice, lattice_type)(**params)


@pytest_cases.fixture
def neighbours_k(reciprocal_lattice):
    return np.array(list(wigner_seitz_neighbours(reciprocal_lattice))) @ reciprocal_lattice.matrix


@pytest_cases.fixture
def gamma_point():
    return np.array([[0., 0., 0.]])


@pytest_cases.fixture
def interior_points(directions, neighbours_k):
    # todo
    pass


@pytest_cases.fixture
def boundary_points(directions, neighbours_k):
    norm2 = (neighbours_k**2).sum(axis=1)
    with np.errstate(divide='ignore'):
        scale = norm2[np.newaxis, :] / np.dot(directions, neighbours_k.T)
    scale[scale < 0] = np.inf
    scale = np.min(scale, axis=1) / 2
    return scale[:, np.newaxis] * directions  # shape (len(directions), 3)

# todo: edge + vertex points?
# todo: stability (+epsilon)?


class TestBackfold:
    @pytest_cases.parametrize("points", [gamma_point, boundary_points])
    def test_translational_invariance(self, reciprocal_lattice, points, neighbours_k):
        # create n_t translationally equivalent points, shape (n_t, n, 3)
        points_t = points + np.vstack([[0, 0, 0], neighbours_k])[:, np.newaxis, :]

        bf = backfold_k(reciprocal_lattice, points_t)  # todo matrix remove for testing

        # assert that backfolded coordinates are equal
        equal = np.all(np.isclose(bf[0], bf), axis=(0, 2))

        # make sure this is the case for all sampled points
        assert all(equal)



class TestBackfoldParallelepiped:
    # test stability
    pass


class TestBackfoldIrred:
    # todo: high symmetry k-points, k-paths
    # for different classes of k-points: (high symmetry points, boundary points, random points etc)
    # high symmetry points: HighSymmKpath(self.primitive_structure).kpath['kpoints']
    # high symmetry path: HighSymmKpath(self.primitive_structure).kpath['kpath']
    # or .get_kpoints(line_density=xxx)
    # boundary:
    pass

