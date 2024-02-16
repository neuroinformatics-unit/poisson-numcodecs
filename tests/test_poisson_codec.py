from poisson_numcodecs.Poisson import Poisson
import numpy as np
import pytest

DEBUG = False

def make_poisson_ramp_signals(shape=(10, 1, 1), min_rate=1, max_rate=5, dtype="int16"):
    assert isinstance(shape, tuple)
    assert len(shape) == 3
    x, y, times = shape
    output_array = np.zeros(shape, dtype=dtype)
    for x_ind in range(x):
        for y_ind in range(y):
            output_array[x_ind, y_ind, :] = np.random.poisson(np.arange(min_rate, max_rate, (max_rate-min_rate)/times))
    return output_array

@pytest.fixture
def test_data(dtype="int16"):
    test2d = make_poisson_ramp_signals(shape=(50, 1, 1), min_rate=1, max_rate=5, dtype=dtype)
    test2d_long = make_poisson_ramp_signals(shape=(1, 50, 1), min_rate=1, max_rate=5, dtype=dtype)

    return [test2d, test2d_long]

def test_poisson_encode_decode(test_data):
    for integer_per_photon in [5, 8, 9, 10]:
        poisson_codec = Poisson(dark_signal=0, signal_to_photon_gain=1.0
                                ,encoded_dtype='int16', decoded_dtype='int16',
                                integer_per_photon=integer_per_photon)
        for example_data in test_data:
            encoded = poisson_codec.encode(example_data)
            decoded = poisson_codec.decode(encoded)
            np.testing.assert_allclose(decoded, example_data, atol=1e-6)

if __name__ == '__main__':
    list_data = test_data("int16")
    test_poisson_encode_decode(list_data)
