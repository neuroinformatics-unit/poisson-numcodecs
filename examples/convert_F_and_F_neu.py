import numpy as np
from poisson_numcodecs import calibrate
import tiffile as tif
from pathlib import Path
import matplotlib.pyplot as plt
import argparse



def convert_movie(movie):
    if np.min(movie) < 0:
        movie = movie + np.min(movie) * -1

    print("Starting the calibration of the movie...")
    calibrator = calibrate.SequentialCalibratePhotons(movie)

    print("Calculate quantal size and intercept...")
    [photon_sensitivity, dark_signal] = calibrator.get_photon_sensitivity_parameters()
    print(f"Quantal size: {photon_sensitivity}\nIntercept: {dark_signal}\n")

    print(
        "Performing the calculation photon flux movie = (scan  - dark_signal) / photon_sensitivity"
    )
    # adapt intercept to the new value
    if dark_signal < 0:
        print("Adapting the intercept to the new value due to negative values in the movie")
        dark_signal = dark_signal - np.min(movie) * -1
    converted_movie = (movie - dark_signal) / photon_sensitivity

    return converted_movie



def simple_neuropil_correction(f: np.ndarray, fneu: np.ndarray) -> np.ndarray:
    """Simple neuropil correction:
    - Subtract 0.7 * Fneu from F

    Parameters
    ----------
    f : np.ndarray
        The fluorescence of the soma
    fneu : np.ndarray
        The fluorescence of the neuropil
    Returns
    -------
    np.ndarray
        The corrected fluorescence
    """
    
    f_corrected = f - fneu * 0.7

    return f_corrected

def plot_f_fneu(f: np.ndarray,
                fneu: np.ndarray,
                f_subtracted: np.ndarray,
                saving_path: Path,
                title: str):
    """Plot the F, Fneu and F - 0.7 * Fneu

    Parameters
    ----------
    f : np.ndarray
        The fluorescence of the soma
    fneu : np.ndarray
        The fluorescence of the neuropil
    f_subtracted : np.ndarray
        The fluorescence of the soma - 0.7 * the fluorescence of the neuropil
    saving_path : Path
        The path to save the plot
    title : str
        The title of the plot
    """    
    plt.plot(f, label='F')
    plt.plot(fneu, label='Fneu')
    plt.plot(f_subtracted, label='F - 0.7 * Fneu')

    plt.legend()
    plt.gcf().set_size_inches(15, 5)
    plt.title(title)

    plt.savefig(saving_path)
    plt.close()


def get_roi_mask(stats_per_roi: dict) -> tuple:
    """Get the mask for the ROI:
    - Exclude overlapping ROIs
    - Keep only pixel considered to be in the soma

    Parameters
    ----------
    stats_per_roi : dict
        The stats for the ROI as calculated by Suite2P

    Returns
    -------
    tuple
        The ypix, xpix and mask for the ROI
    """
    ypix = stats_per_roi['ypix']
    xpix = stats_per_roi['xpix']
    overlap = stats_per_roi['overlap'] # exclude overlapping ROIs
    soma_crop = stats_per_roi['soma_crop'] # keep only pixel considered to be in the soma

    mask = ~overlap & soma_crop
    ypix = ypix[mask]
    xpix = xpix[mask]

    return ypix, xpix, mask

def load_suite2p_data(suite2p_path: Path) -> tuple:
    """Load the Suite2P data:
    - Load the F and Fneu
    - Load the stats
    - Load the registered movie

    Parameters
    ----------
    suite2p_path : Path
        The path to the Suite2P folder

    Returns
    -------
    tuple
        The F, Fneu, stats and registered movie
    """    
    f_neu_path = Path(suite2p_path) / 'plane0/Fneu.npy'
    f_path = Path(suite2p_path) / 'plane0/F.npy'
    suite2p_stats_path = Path(suite2p_path) / 'plane0/stat.npy'
    path_to_bin_file = Path(suite2p_path) / 'plane0/data.bin'
    ops_path = Path(suite2p_path) / 'plane0/ops.npy'

    ops = np.load(ops_path, allow_pickle=True).item()

    shape_image = ( ops["nframes"], ops['Ly'], ops['Lx'])
    registered_movie = np.memmap(
        path_to_bin_file, shape=shape_image, dtype="int16"
    )

    stats = np.load(suite2p_stats_path, allow_pickle=True)
    f = np.load(f_path)
    fneu = np.load(f_neu_path)

    return f, fneu, stats, registered_movie

def get_photon_flux_f_fneu(stats_per_roi: dict, converted_movie_registered: np.ndarray) -> tuple:
    """Get the photon flux F and Fneu for a given ROI:
    - Get the mask for the ROI
    - Recalculate the weighted fluorescence using the photon converted movie
    - Calculate the neuropil fluorescence using the neuropil mask

    Parameters
    ----------
    stats_per_roi : dict
        The stats for the ROI as calculated by Suite2P
    converted_movie_registered : np.ndarray
        The registered movie converted to photon flux

    Returns
    -------
    tuple
        The photon flux F and Fneu
    """    
    # f is a weighted average and fneu is a simple average over pixels
    # see https://github.com/MouseLand/suite2p/blob/main/suite2p/extraction/extract.py

    # we need first to get the mask for the ROI
    ypix, xpix, boolean_mask = get_roi_mask(stats_per_roi)
    # then we recalculate the weighted fluorescence using the photon converted movie
    # lam is the weight of each pixel in the ROI
    f_photon_flux = np.dot(converted_movie_registered[:, ypix, xpix], stats_per_roi['lam'][boolean_mask]) / len(boolean_mask)

    # neuropil mask 
    neuropil_mask = stats_per_roi["neuropil_mask"]
    fneu_n_pixels = len(neuropil_mask)
    # we can directly convert the neuropil fluorescence
    fneu_photon_flux = np.asarray([
        frame.flatten()[neuropil_mask].sum() / fneu_n_pixels
        for frame in converted_movie_registered
    ])

    return f_photon_flux, fneu_photon_flux

def save_photon_flux_f_fneu(
    f: np.ndarray,
    fneu: np.ndarray,
    saving_path: Path):
    """Save the photon flux F and Fneu

    Parameters
    ----------
    f : np.ndarray
        The photon flux F
    fneu : np.ndarray
        The photon flux Fneu
    saving_path : Path
        The path to save the photon flux F and Fneu
    """    

    np.save(saving_path / "f_corrected.npy", f)
    np.save(saving_path / "fneu_corrected.npy", fneu)

    print(f"Saved photon flux F and Fneu in {saving_path}")

def main(tif_path: Path, suite_2p_path: Path, make_plots: bool=False):
    """Main function to convert F and Fneu to photon flux:
    - Convert the original movie to photon flux for comparison
    - Load the suite2p data
    - Convert the registered movie to photon flux
    - Calculate the photon flux F and Fneu
    - Make plots if needed
    - Save the photon flux F and Fneu

    Parameters
    ----------
    tif_path : Path
        The path to the original TIFF file
    suite_2p_path : Path
        The path to the Suite2P folder with analized data
    make_plots : bool, optional
        Whether to make plots or not, by default False
    """
    #  calculate on the raw movie as a meter of comparison
    print("Converting the original movie to photon flux...")
    tiff = tif.imread(tif_path)
    convert_movie(tiff)

    #  load the suite2p data
    f, fneu, stats, registered_movie = load_suite2p_data(suite_2p_path)

    # convert to photon flux the registered movie 
    # quantal size and intercept are recalculated, they might differ from the original movie
    print("Recalculating the photon sensitivity and dark signal for the registered movie...")
    print("Please compare the printed qunal size and intercept with the original movie")
    converted_movie_registered = convert_movie(registered_movie)
    
    # for comparison
    f_subtracted = simple_neuropil_correction(f, fneu)

    Fs, Fneus = [], []
    for roi_id, stats_per_roi in enumerate(stats):
        f_photon_flux, fneu_photon_flux = get_photon_flux_f_fneu(stats_per_roi, converted_movie_registered)
        Fs.append(f_photon_flux)
        Fneus.append(fneu_photon_flux)
        
        f_subtracted_flux = simple_neuropil_correction(f_photon_flux, fneu_photon_flux)

        if make_plots:
            plot_f_fneu(f[roi_id], fneu[roi_id], f_subtracted[roi_id], f'{tif_path.parent}/new_original_f_fneu_roi_{roi_id}.png', 'Original neuropil decontamination [a.u.]')
            plot_f_fneu(f_photon_flux, fneu_photon_flux, f_subtracted_flux,
                        f'{tif_path.parent}/new_photon_flux_f_fneu_roi_{roi_id}.png', 'Photon flux neuropil decontamination [photons/pixel/roi]')

    save_photon_flux_f_fneu(Fs, Fneus, Path(suite_2p_path) / 'plane0/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("tif_path", type=Path, help="Path to the TIFF file")
    parser.add_argument("suite_2p_path", type=Path, help="Path to the Suite2P folder")
    parser.add_argument("--make_plots", help="Whether to make plots or not", default=False)

    args = parser.parse_args()

    main(args.tif_path, args.suite_2p_path, args.make_plots)