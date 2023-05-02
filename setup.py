# -*- coding: utf-8 -*-
from os import environ
from setuptools import setup, Extension
import numpy as np

compile_args = ['-O3', '-ffast-math']
link_args = []

if not 'BUILD_PLATFORM_INDEPENDENT' in environ:
    #non-agnostic build
    compile_args.append('-march=native')

if not 'BUILD_NO_OPENMP' in environ:
    #openmp support
    compile_args.append('-fopenmp')
    link_args.append('-fopenmp')

setup(
    package_dir={'': 'src'},  # so that `python setup.py build_ext --inplace` works
    ext_modules=[
        Extension('fplore.fast_util', sources=['src/fplore/fast_util.pyx'],
                  include_dirs=[np.get_include()],
                  extra_compile_args=compile_args,
                  extra_link_args=link_args),
    ],
)
