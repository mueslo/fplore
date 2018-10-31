import numpy as np


def cartesian_product(*xs):
    """Iterates over primary axis first, then second, etc."""
    return np.array(np.meshgrid(*xs, indexing='ij')).reshape(len(xs), -1).T

