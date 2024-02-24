from poisson_numcodecs.Calibrate import RasterCalibratePhotons
import numpy as np
import pytest

DEBUG = False

def make_fake_movie(gain=10, offset=1, shape=(100, 10, 10), min_rate=1, max_rate=5, dtype="int16"):
    assert isinstance(shape, tuple)
    assert len(shape) == 3
    times, x, y = shape
    output_array = np.zeros(shape, dtype=dtype)
    for x_ind in range(x):
        for y_ind in range(y):
            # We pick a random rate for each pixel
            rate = np.random.uniform(min_rate, max_rate)
            output_array[:, x_ind, y_ind] = gain*np.random.poisson(rate, size=times)+offset
    return output_array


def test_gain_extraction():
    for fake_gain in [10, 20]:
        for fake_offset in [1, 2]:
            test2d = make_fake_movie(gain=fake_gain, offset=fake_offset, shape=(1000, 256, 256))
            calibrator = RasterCalibratePhotons(test2d)
            [photon_gain, photon_offset]=calibrator.get_photon_gain_parameters(perc_min=0, perc_max=100)
            print(photon_gain, photon_offset)

            # We check that the gain and offset are within X% of the true value
            np.testing.assert_allclose(photon_gain[0], fake_gain, atol=fake_gain/20)
            np.testing.assert_allclose(-photon_offset[0]/photon_gain[0], fake_offset, atol=fake_offset/20)

if __name__ == '__main__':
    test_gain_extraction()
