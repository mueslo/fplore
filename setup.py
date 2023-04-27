# -*- coding: utf-8 -*-
from setuptools import setup, Extension


setup(
    ext_modules=[
        Extension('fplore.fast_util', sources=['fplore/fast_util.pyx']),
    ],
)
