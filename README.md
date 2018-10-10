# fplore
FPLO run evaluation

## Installation

blah blah virtualenv pip

## Usage

There are various ways of accessing your FPLO run data within fplore.

1. Access the processed data in the scope of an FPLO run

2. Access the raw data directly


### Examples

#### Usage method 1 (data access via FPLO run)

This will automatically load band data from `+band` or `+band_kp`, whichever is available. It will also process all *k*-points, so they lie within the first Brillouin zone.

```
from fplore.loader import FPLORun
run = FPLORun("/home/jdoe/fplo_run/")
band_data = run.band_data
```

#### Usage method 2 (direct data access)

Alternatively, you may want to use the raw data as it appears in FPLO's written files without any convenience functionality.

```
from fplore.loader import FPLORun
run = FPLORun("/home/jdoe/fplo_run/")
raw_band_data = run['+band'].data
```

However, this will fail if you are using non-standard filenames. In that case you can manually load a file by using the appropriate loader class directly.

```
from fplore.loader import Band
band = Band("/home/jdoe/fplo_run/+band_old")
raw_band_data = band.data
```