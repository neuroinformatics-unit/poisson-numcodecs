from poisson_numcodecs.calibrate import RasterCalibratePhotons, SequentialCalibratePhotons
import numpy as np
import pytest

DEBUG = False
SEED = 7916
rng = np.random.default_rng(SEED)

def make_fake_movie(gains=[10], offsets=[1], shape=(100, 10, 10), min_rate=1, max_rate=5, dtype="int16"):
    assert isinstance(shape, tuple)
    assert len(shape) == 3
    times, x, y = shape
    output_array = np.zeros(shape, dtype=dtype)

    nb_gains = len(gains)

    for x_ind in range(x):
        for y_ind in range(y):
            # We pick a random rate for each pixel
            rate = rng.uniform(min_rate, max_rate)

            # We pick a random gain and offset for each pixel
            index_param = rng.integers(0, nb_gains)
            gain = gains[index_param]
            offset = offsets[index_param]

            output_array[:, x_ind, y_ind] = gain*rng.poisson(rate, size=times)+offset
    return output_array

def test_single_gain_extraction():
    for fake_gain in [10, 20]:
        for fake_offset in [1, 2]:
            test2d = make_fake_movie(gains=[fake_gain], offsets=[fake_offset], shape=(1000, 256, 256))
            calibrator = RasterCalibratePhotons(test2d)
            [photon_sensitivity, dark_signal]=calibrator.get_photon_sensitivity_parameters(perc_min=0, perc_max=100)
            print(photon_sensitivity, dark_signal)

            # We check that the gain and offset are within X% of the true value
            np.testing.assert_allclose(photon_sensitivity[0], fake_gain, atol=fake_gain/20)
            np.testing.assert_allclose(dark_signal[0], fake_offset, atol=fake_offset/20)

def test_multi_gain_extraction():
    test2d = make_fake_movie(gains=[10, 15], offsets=[1, 2], shape=(1000, 256, 256))
    calibrator = RasterCalibratePhotons(test2d)
    [photon_sensitivitys, dark_signals]=calibrator.get_photon_sensitivity_parameters(n_groups=2, perc_min=0, perc_max=100)
    print(photon_sensitivitys, dark_signals)

    # We permute the estimated gain and offset to match the ground truth
    photon_sensitivitys = np.sort(photon_sensitivitys)
    dark_signals = np.sort(dark_signals)

    np.testing.assert_allclose(photon_sensitivitys, [10, 15], atol=1)

def test_sequential_gain_extraction():
    fake_gain = 2.5
    fake_offset = 1

    test2d = make_fake_movie(gains=[fake_gain], offsets=[fake_offset], shape=(1000, 256, 256), min_rate=1, max_rate=50)
    calibrator = SequentialCalibratePhotons(test2d)
    [photon_sensitivity, dark_signal]=calibrator.get_photon_sensitivity_parameters()

    np.testing.assert_allclose(photon_sensitivity, fake_gain, atol=fake_gain/20)
    np.testing.assert_allclose(dark_signal, fake_offset, atol=fake_offset/20)

def test_agreement_calibrators():
    fake_gain = 2.5
    fake_offset = 10

    test2d = make_fake_movie(gains=[fake_gain], offsets=[fake_offset], shape=(2000, 256, 256), min_rate=1, max_rate=50)
    calibrator_raster = RasterCalibratePhotons(test2d)
    calibrator_sequential = SequentialCalibratePhotons(test2d)
    [photon_sensitivity_raster, dark_signal_raster]=calibrator_raster.get_photon_sensitivity_parameters()
    [photon_sensitivity_sequential, dark_signal_sequential]=calibrator_sequential.get_photon_sensitivity_parameters()

    np.testing.assert_allclose(photon_sensitivity_raster[0], photon_sensitivity_sequential, atol=fake_gain/20)
    np.testing.assert_allclose(dark_signal_raster[0], dark_signal_sequential, atol=fake_offset/20)


if __name__ == '__main__':
    test_single_gain_extraction()
    test_multi_gain_extraction()
    test_sequential_gain_extraction()