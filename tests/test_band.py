import pytest
import numpy as np
from fplore.loader import Band


def test_reshape_regular():
    # asserts that reshaped data is equal to meshgrid(indexing='ij') output
    # (among other things so it can be directly used for interpolation input)

    # construct Band data (todo: fixture)
    x, y, z = np.linspace(-0.5, -0.25, 43), np.linspace(-0.25, 0.,
                                                        47), np.linspace(0, 0.5,
                                                                         53)
    e = lambda x, y, z: np.array(
        [x, y, z, x ** 2 - y ** 2 + z, y ** 2 - x ** 2 + z])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    E = e(X, Y, Z).transpose(1, 2, 3, 0)
    E_list = E.reshape(-1, 5)

    bd = Band._gen_band_data_array(*E_list.shape)

    bd['k'] = E_list[:, :3]
    bd['e'] = E_list
    np.random.shuffle(bd)

    band = Band('fake_file')
    band.data = bd
    band.is_loaded = True

    # function to test
    (xr, yr, zr), bdr = band.reshaped_data

    # test
    assert tuple(map(len, (xr, yr, zr))) == tuple(map(len, (x, y, z)))
    assert np.allclose(xr, x)
    assert np.allclose(yr, y)
    assert np.allclose(zr, z)

    assert bdr['e'].shape == E.shape
    assert np.allclose(bdr['e'], E)
