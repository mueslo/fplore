.. fplore documentation master file, created by
   sphinx-quickstart on Tue Oct 23 01:47:15 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to fplore's documentation!
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. automodule:: fplore
   :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
* :doc:`Example gallery <gallery/index>`

Introduction
============

This library is meant to help make data evaluation of `FPLO`_ runs easier by supplying easy ways to load FPLO data and by offering some convenience functionality.

You can take a look at some examples :doc:`here <gallery/index>`.

There are various ways of accessing your FPLO run data within fplore.

1. Load the data in the scope of an FPLO run

2. Load the data directly

Usually, number one would be preferable, as it would for example allow you to apply the symmetry operations from the symmetry group stated in the =.in file automatically to other data.

.. _FPLO: https://www.fplo.de/

Usage method 1
--------------

With this method we will specify the FPLO run directory.

For example, the following will automatically load band data from ``+band`` or ``+band_kp``, whichever is available. It will also process all *k*-points, so they lie within the first Brillouin zone.

.. code-block:: python

   from fplore import FPLORun
   run = FPLORun("/home/jdoe/fplo_run/")
   band_data = run.band.data

You can also specify the filename directly.

.. code-block:: python

   band_data = run['+band_kp'].data

For example, you can access arbitrary configuration settings from your ``=.in`` like so:

.. code-block:: python

   In [3]: run["=.in"].sections
   Out[3]: dict_keys(['header', 'structure_information', 'structure_dependend', 'mesh', 'brillouin', 'bandstructure_plot', 'iteration_control', 'forces', 'options', 'cpa', 'numerics', 'LSDA_U', 'OPC', 'Advanced_output'])

   In [4]: run["=.in"].structure_information
   Out[4]: {'structure_type': {'type': 1, 'description': 'Crystal'}, 'spacegroup': {'number': 139, 'symbol': 'I4/MMM'}, 'subgroupgenerators': [], 'lengthunit': {'type': 2, 'description': 'Angstroem'}, 'lattice_constants': [4.0345, 4.0345, 9.828], 'axis_angles': [90.0, 90.0, 90.0], 'max_L': 4, 'nsort': 3, 'wyckoff_positions': [{'element': 'Yb', 'tau': [0, 0, 0]}, {'element': 'Ir', 'tau': [Fraction(1, 2), 0, Fraction(-1, 4)]}, {'element': 'Si', 'tau': [0, 0, 0.3807989986]}]}

   In [5]: run["=.in"].structure_information.lengthunit.description
   Out[5]: 'Angstroem'

Usage method 2
--------------

If you just want to use ``fplore`` to load the raw data of some file and nothing more, you can manually load a file by running

.. code-block:: python

   from fplore.files import FPLOFile
   band = FPLOFile.load("/home/jdoe/fplo_run/+band)
   raw_band_data = band.data

Which will automatically pick the correct loader class based on the filename. If the files are renamed and this fails, you can manually specify the loader class. See :py:attr:`fplore.files.base.FPLOFile.registry` for the appropriate loader class. For example:

.. code-block:: python

   from fplore.files import Band
   band = Band("/home/jdoe/fplo_run/+band_old")
   raw_band_data = band.data

Some ``FPLORun`` properties (excerpt)
-------------------------------------

.. code-block:: python

   In [1]: run.spacegroup_number
   Out[1]: 139
   In [2]: run.spacegroup
   Out[2]: <pymatgen.symmetry.groups.cached_class.<locals>._decorated object at 0x7ff1609ca668>
   In [3]: run.lattice
   Out[3]: Lattice
       abc : 4.0345 4.0345 9.828
    angles : 90.0 90.0 90.0
    volume : 159.97222577700003
         A : 4.0345 0.0 2.4704187555799985e-16
         B : 6.48797083012039e-16 4.0345 2.4704187555799985e-16
         C : 0.0 0.0 9.828
   In [4]: run.structure
   Out[4]: Structure Summary
   Lattice
       abc : 4.0345 4.0345 9.828
    angles : 90.0 90.0 90.0
    volume : 159.97222577700003
         A : 4.0345 0.0 2.4704187555799985e-16
         B : 6.48797083012039e-16 4.0345 2.4704187555799985e-16
         C : 0.0 0.0 9.828
   PeriodicSite: Yb (0.0000, 0.0000, 0.0000) [0.0000, 0.0000, 0.0000]
   PeriodicSite: Yb (2.0173, 2.0173, 4.9140) [0.5000, 0.5000, 0.5000]
   PeriodicSite: Ir (2.0173, 0.0000, 7.3710) [0.5000, 0.0000, 0.7500]
   PeriodicSite: Ir (0.0000, 2.0173, 7.3710) [0.0000, 0.5000, 0.7500]
   PeriodicSite: Ir (2.0173, 0.0000, 2.4570) [0.5000, 0.0000, 0.2500]
   PeriodicSite: Ir (0.0000, 2.0173, 2.4570) [0.0000, 0.5000, 0.2500]
   PeriodicSite: Si (0.0000, 0.0000, 3.7425) [0.0000, 0.0000, 0.3808]
   PeriodicSite: Si (0.0000, 0.0000, 6.0855) [0.0000, 0.0000, 0.6192]
   PeriodicSite: Si (2.0173, 2.0173, 8.6565) [0.5000, 0.5000, 0.8808]
   PeriodicSite: Si (2.0173, 2.0173, 1.1715) [0.5000, 0.5000, 0.1192]

Running the examples
====================

You can run the :doc:`examples <gallery/index>` in the ``examples`` subdirectory by running them as a module, for example ``python -m examples.band_plot``.
