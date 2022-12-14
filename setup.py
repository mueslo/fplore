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
    long_description=open('README.md', 'r').read(),
    long_description_content_type='text/markdown',
    license='GPLv3',
    url='https://github.com/mueslo/fplore',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=[
        "numpy~=1.21",
        "scipy~=1.7",
        "pymatgen~=2022.5.19",
        "orderedattrdict~=1.5",
        "logbook~=1.5",
        "pyparsing~=2.4",
        "progressbar2~=3.38",
        "matplotlib~=3.4,<3.6",
        "Pint~=0.17",
    ],
    python_requires=">=3.8, <4",  # see also docs, test, ci config
    extras_require={
        'tests': ['tox'],
        'docs': [
            'sphinx~=4.5',
            'sphinx-gallery~=0.10',
            'Pillow',
        ],
    },
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
)
