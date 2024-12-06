import numpy as np
from poisson_numcodecs import calibrate
import tiffile as tif
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

def get_adapted_parameters(input_path=None):
    scan = tif.imread(input_path)
    shift = np.min(scan)
    if np.min(scan) < 0:
        scan = scan + np.abs(shift)

    
    print("Converting movie to photon flux movie...")
    calibrator = calibrate.SequentialCalibratePhotons(scan)

    [photon_sensitivity, dark_signal] = calibrator.get_photon_sensitivity_parameters()
    print(f"Quantal size: {photon_sensitivity}\nIntercept: {dark_signal}\n")

    dark_signal = dark_signal - np.abs(shift)
    print(f"Adapted intercept: {dark_signal}")

    return photon_sensitivity, dark_signal

def read_f_fneu(f_path=None, fneu_path=None):
    f = np.load(f_path)
    fneu = np.load(fneu_path)

    return f, fneu

def convert_f_fneu_to_photon_flux(f=None, fneu=None, photon_sensitivity=None, dark_signal=None):
    f_photon_flux = (f - dark_signal) / photon_sensitivity
    fneu_photon_flux = (fneu - dark_signal) / photon_sensitivity

    return f_photon_flux, fneu_photon_flux

def simple_neuropil_correction(f=None, fneu=None):
    f_corrected = f - fneu * 0.7

    return f_corrected

def plot_f_fneu(f=None, fneu=None, f_subtracted=None, saving_path=None, title=None):
    plt.plot(f, label='F')
    plt.plot(fneu, label='Fneu')
    plt.plot(f_subtracted, label='F - 0.7 * Fneu')
    plt.legend()
    #  set figure size
    plt.gcf().set_size_inches(15, 5)
    plt.title(title)
    plt.savefig(saving_path)
    plt.close()

def main(tif_path, suite_2p_path):

    photon_sensitivity, dark_signal = get_adapted_parameters(tif_path)
    f_neu_path = Path(suite_2p_path) / 'plane0/Fneu.npy'
    f_path = Path(suite_2p_path) / 'plane0/F.npy'
    f, fneu = read_f_fneu(f_path, f_neu_path)
    f_subtracted = simple_neuropil_correction(f, fneu)
    f_photon_flux, fneu_photon_flux = convert_f_fneu_to_photon_flux(f, fneu, photon_sensitivity, dark_signal)
    f_subtracted_flux = simple_neuropil_correction(f_photon_flux, fneu_photon_flux)

    roi_id = 18
    plot_f_fneu(f[roi_id], fneu[roi_id], f_subtracted[roi_id], 'figures/original_f_fneu.png', 'Original neuropil decontamination [a.u.]')
    plot_f_fneu(f_photon_flux[roi_id], fneu_photon_flux[roi_id], f_subtracted_flux[roi_id],
                'figures/photon_flux_f_fneu.png', 'Photon flux neuropil decontamination [photons/s]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("tif_path", type=str, help="Path to the TIFF file")
    parser.add_argument("suite_2p_path", type=str, help="Path to the Suite2P folder")

    args = parser.parse_args()

    main(args.tif_path, args.suite_2p_path)