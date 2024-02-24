from poisson_numcodecs.Calibrate import RasterCalibratePhotons
import numpy as np
import pytest

DEBUG = False

def make_fake_movie(gains=[10], offsets=[1], shape=(100, 10, 10), min_rate=1, max_rate=5, dtype="int16"):
    assert isinstance(shape, tuple)
    assert len(shape) == 3
    times, x, y = shape
    output_array = np.zeros(shape, dtype=dtype)

    nb_gains = len(gains)

    for x_ind in range(x):
        for y_ind in range(y):
            # We pick a random rate for each pixel
            rate = np.random.uniform(min_rate, max_rate)

            # We pick a random gain and offset for each pixel
            gain = gains[np.random.randint(0, nb_gains)]
            offset = offsets[np.random.randint(0, nb_gains)]

            output_array[:, x_ind, y_ind] = gain*np.random.poisson(rate, size=times)+offset
    return output_array


def test_single_gain_extraction():
    for fake_gain in [10, 20]:
        for fake_offset in [1, 2]:
            test2d = make_fake_movie(gains=[fake_gain], offsets=[fake_offset], shape=(1000, 256, 256))
            calibrator = RasterCalibratePhotons(test2d)
            [photon_gain, photon_offset]=calibrator.get_photon_gain_parameters(perc_min=0, perc_max=100)
            print(photon_gain, photon_offset)

            # We check that the gain and offset are within X% of the true value
            np.testing.assert_allclose(photon_gain[0], fake_gain, atol=fake_gain/20)
            np.testing.assert_allclose(-photon_offset[0]/photon_gain[0], fake_offset, atol=fake_offset/20)

def test_multi_gain_extraction():
    test2d = make_fake_movie(gains=[10, 15], offsets=[1, 2], shape=(1000, 256, 256))
    calibrator = RasterCalibratePhotons(test2d)
    [photon_gains, photon_offsets]=calibrator.get_photon_gain_parameters(n_groups=2, perc_min=0, perc_max=100)
    print(photon_gains, photon_offsets)

    # We permute the estimated gain and offset to match the ground truth
    photon_gains = np.sort(photon_gains)
    photon_offsets = np.sort(photon_offsets)

    np.testing.assert_allclose(photon_gains, [10, 15], atol=1)

if __name__ == '__main__':
    test_single_gain_extraction()
    test_multi_gain_extraction()
