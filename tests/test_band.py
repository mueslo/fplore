import numpy as np
from fplore.files import Band
from fplore.util import cartesian_product


def test_reshape_regular():
    # asserts that reshaped data is equal to meshgrid(indexing='ij') output
    # (among other things so it can be directly used for interpolation input)

    # construct Band data (todo: fixture)
    kx, ky, kz = (np.linspace(-0.5, 0, 43), np.linspace(-0.25, 0., 47),
               np.linspace(0, 0.5, 53))

    coords = cartesian_product(kx, ky, kz)

    bd = Band._gen_band_data_array(len(coords), k_coords=True, index=True)

    bd['k'] = coords
    bd['idx'] = np.arange(len(coords))
    np.random.shuffle(bd)

    band = Band('fake_file')
    band.data = band.symm_data = band.padded_symm_data = bd
    band.is_loaded = True

    # function to test
    (xr, yr, zr), bdr = band.reshape_gridded_data()

    # test
    assert tuple(map(len, (xr, yr, zr))) == tuple(map(len, (kx, ky, kz)))
    assert np.allclose(xr, kx)
    assert np.allclose(yr, ky)
    assert np.allclose(zr, kz)
