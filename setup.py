# -*- coding: utf-8 -*-
from setuptools import setup, Extension
import numpy as np

setup(
    ext_modules=[
        Extension('fplore.fast_util', sources=['src/fplore/fast_util.pyx'],
                  include_dirs=[np.get_include()]),
    ],
)
