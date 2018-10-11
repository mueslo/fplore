# fplore
A python library for FPLO run evaluation.

:construction: Still a work in progress, syntax may change without notice. 

:raising_hand: If you have completed FPLO runs, It'd be super helpful if you could send them to me so I can test various cases and implement new features. Thanks!

## Installation

blah blah virtualenv pip

## Usage

There are various ways of accessing your FPLO run data within fplore.

1. Access the processed data in the scope of an FPLO run

2. Access the raw data directly


### Examples

You can run the examples in the `examples` subdirectory by running them as a module, for example `python -m examples.band_plot`.

#### Usage method 1 (data access via FPLO run)

This will automatically load band data from `+band` or `+band_kp`, whichever is available. It will also process all *k*-points, so they lie within the first Brillouin zone.

```python
from fplore.loader import FPLORun
run = FPLORun("/home/jdoe/fplo_run/")
band_data = run.band_data
```

#### Usage method 2 (direct data access)

Alternatively, you may want to use the raw data as it appears in FPLO's written files without any convenience functionality.

```python
from fplore.loader import FPLORun
run = FPLORun("/home/jdoe/fplo_run/")
raw_band_data = run['+band'].data
```

For example, you can access arbitrary configuration settings from your `=.in` like so:

```
>>> dict(run["=.in"].bandstructure_plot.bandplot_control)
{'bandplot': True, 'read_sympoints': True, 'ndivisions': 50, 'emin': -2, 'emax': 2, 'nptdos': 1000, 'plot_idos': False, 'plot_ndos': False, 'restrict_bands_to_window': False, 'coeffout': False}
```

However, this will fail if you are using non-standard filenames. In that case you can manually load a file by using the appropriate loader class directly. For example:

```python
from fplore.loader import Band
band = Band("/home/jdoe/fplo_run/+band_old")
raw_band_data = band.data
```

#### `FPLORun` properties

```
>>> run.spacegroup_number
139
>>> run.spacegroup
<pymatgen.symmetry.groups.cached_class.<locals>._decorated object at 0x7ff1609ca668>
>>> run.lattice
Lattice
    abc : 4.0345 4.0345 9.828
 angles : 90.0 90.0 90.0
 volume : 159.97222577700003
      A : 4.0345 0.0 2.4704187555799985e-16
      B : 6.48797083012039e-16 4.0345 2.4704187555799985e-16
      C : 0.0 0.0 9.828
>>> run.structure
Structure Summary
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
```