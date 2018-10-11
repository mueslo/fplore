# -*- coding: utf-8 -*-
from setuptools import setup, find_packages
import codecs
import re
from os import path

here = path.abspath(path.dirname(__file__))


def read(*parts):
    with codecs.open(path.join(here, *parts), 'r') as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


setup(
    name='fplore',
    version=find_version("fplore", "__init__.py"),
    description="FPLO run evaluation",
    license='GPLv3',
    url='https://github.com/mueslo/fplore',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=[
        "six~=1.11",
        "numpy~=1.15",
        "scipy~=1.1",
        "pymatgen~=2018.9.30",
        "orderedattrdict~=1.5",
        "logbook~=1.4",
        "pyparsing~=2.2",
        "progressbar2~=3.38",
        "matplotlib~=3.0",
    ],
    python_requires="~=2.7",
    tests_require=['pytest', 'pytest-logbook'],
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    #entry_points={
    #    'console_scripts': [
    #        'scriptname = fplore.__main__:main',
    #    ]
    #},
)
