from delta2D_numcodecs import Delta2D
import numpy as np
import zarr
import pytest

DEBUG = False

dtypes = ["int8", "int16", "int32", "float32"]

def run_all_options(data):
    dtype = data.dtype
    for axis in (0, 1):
        print(f"Dtype {dtype} - axis {axis}")

        cod = Delta2D(dtype=dtype, num_channels=data.shape[1], axis=axis)
        enc = cod.encode(data)
        assert enc.shape == data.shape

        if axis == 0:
            np.testing.assert_array_equal(enc[0], data[0])
            np.testing.assert_array_equal(enc[1:, :], np.diff(data, axis=axis))
        else:
            np.testing.assert_array_equal(enc[:, 0], data[:, 0])
            np.testing.assert_array_equal(enc[:, 1:], np.diff(data, axis=axis))
        
        dec = cod.decode(enc)

        if dtype.kind != "f":
            np.testing.assert_array_equal(dec, data)
        else:
            np.testing.assert_array_almost_equal(dec, data, decimal=3)
        

def make_noisy_sin_signals(shape=(30000,), sin_f=100, sin_amp=50, noise_amp=5,
                           sample_rate=30000, dtype="int16"):
    assert isinstance(shape, tuple)
    assert len(shape) <= 3
    if len(shape) == 1:
        y = np.sin(2 * np.pi * sin_f * np.arange(shape[0]) / sample_rate) * sin_amp
        y = y + np.random.randn(shape[0]) * noise_amp
        y = y.astype(dtype)
    elif len(shape) == 2:
        nsamples, nchannels = shape
        y = np.zeros(shape, dtype=dtype)
        for ch in range(nchannels):
            y[:, ch] = make_noisy_sin_signals((nsamples,), sin_f, sin_amp, noise_amp,
                                              sample_rate, dtype)
    else:
        nsamples, nchannels1, nchannels2 = shape
        y = np.zeros(shape, dtype=dtype)
        for ch1 in range(nchannels1):
            for ch2 in range(nchannels2):
                y[:, ch1, ch2] = make_noisy_sin_signals((nsamples,), sin_f, sin_amp, noise_amp,
                                                        sample_rate, dtype)
    return y


def generate_test_signals(dtype):
    test2d = make_noisy_sin_signals(shape=(3000, 10), dtype=dtype)
    test2d_long = make_noisy_sin_signals(shape=(200000, 20), dtype=dtype)

    return [test2d, test2d_long]

@pytest.mark.numcodecs
def test_delta2D_numcodecs():
    for dtype in dtypes:
        print(f"\n\nNUMCODECS: testing dtype {dtype}\n\n")

        test_signals = generate_test_signals(dtype)

        for test_sig in test_signals:
            print(f"signal shape: {test_sig.shape}")
            run_all_options(test_sig)

@pytest.mark.zarr
def test_delta2D_zarr():
    for dtype in dtypes:
        print(f"\n\nZARR: testing dtype {dtype}\n\n")
        test_signals = generate_test_signals(dtype)

        
        for test_sig in test_signals:
            for channel_block_size in (3, None):
                print(f"signal shape: {test_sig.shape}")
                num_channels = channel_block_size if channel_block_size is not None else test_sig.shape[1]
                delta0 = Delta2D(dtype=dtype, num_channels=num_channels, axis=0)
                delta1 = Delta2D(dtype=dtype, num_channels=num_channels, axis=1)
                for filters in [[delta0], [delta1], [delta0, delta1], [delta1, delta0]]:
                    print(f"filters: {filters}")
                
                    z = zarr.array(test_sig, chunks=(None, channel_block_size), filters=filters)
                    assert z[:].shape == test_sig.shape
                    assert z[:100, :10].shape == test_sig[:100, :10].shape

                    z = zarr.array(test_sig, chunks=(1000, channel_block_size), filters=filters)
                    assert z[:].shape == test_sig.shape
                    assert z[:100, :10].shape == test_sig[:100, :10].shape


if __name__ == '__main__':
    test_delta2D_numcodecs()
    test_delta2D_zarr()
