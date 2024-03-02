import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import HuberRegressor as Regressor

class CalibratePhotons():
    def __init__(self, data_array_movie):

        # We first check we have a 3D movie
        if len(data_array_movie.shape) != 3:
            raise ValueError("The data array movie should be N x Y x X. where N is the number of frames and Y and X are the spatial dimensions.")
        
        self.data_array_movie = data_array_movie
        
        self.std_image = None
        self.mean_image = None
        self.photon_sensitivity = None
        self.dark_signal = None

    def get_mean_image(self):
        """Get the mean image of the data array movie."""
        if self.mean_image is None:
            self.mean_image = np.mean(self.data_array_movie, axis=0)
        return self.mean_image
    
    def get_std_image(self):
        """Get the standard deviation image of the data array movie."""
        if self.std_image is None:
            self.std_image = np.std(self.data_array_movie, axis=0)
        return self.std_image
    
    def plot_std_projection_image(self, min_range=1, max_range=99):
        """Plot the standard deviation projection image."""
        fig = plt.figure()
        image_project = self.get_std_image()

        list_pixel_limits = np.percentile(image_project.flatten(), [min_range, max_range])
        plt.imshow(image_project, cmap="gray", vmin=list_pixel_limits[0], vmax=list_pixel_limits[1], interpolation='none')
        plt.colorbar()
        plt.axis("off")
        return fig
    
    def plot_mean_projection_image(self, min_range=1, max_range=99):
        """Plot the mean projection image."""
        fig = plt.figure()
        image_project = self.get_mean_image()

        list_pixel_limits = np.percentile(image_project.flatten(), [min_range, max_range])
        plt.imshow(image_project, cmap="gray", vmin=list_pixel_limits[0], vmax=list_pixel_limits[1], interpolation='none')
        plt.colorbar()
        plt.axis("off")
        return fig
    
    def subsample_and_crop_video(self, crop, start_frame=0, end_frame=-1):
        """Subsample and crop a video, cache results. Also functions as a data_pointer load.

        Args:
            crop:  A tuple (px_y, px_x) specifying the number of pixels to remove
            start_frame:  The index of the first desired frame
            end_frame:  The index of the last desired frame

        Returns:
            The resultant array.
        """

        # We first reset the saved data
        self.mean_image = None
        self.std_image = None
        self.photon_sensitivity = None
        self.dark_signal = None
        
        _shape = self.data_array_movie.shape
        px_y_start, px_x_start = crop
        px_y_end = _shape[1] - px_y_start
        px_x_end = _shape[2] - px_x_start

        if start_frame == _shape[0] - 1 and (end_frame == -1 or end_frame == _shape[0]):
            cropped_video = self.data_array_movie[
                start_frame:_shape[0], px_y_start:px_y_end, px_x_start:px_x_end
            ]
        else:
            cropped_video = self.data_array_movie[
                start_frame:end_frame, px_y_start:px_y_end, px_x_start:px_x_end
            ]
        self.data_array_movie = cropped_video

    def get_photon_flux_movie(self):        
        """Get the photon flux movie. This is the movie with the photon gain and offset applied."""
        if self.photon_sensitivity is None:
            raise ValueError("You need to compute the photon gain parameters first.")
        else:
            photon_flux = (self.data_array_movie
                        .astype('float') - self.dark_signalstype('float')) / self.photon_sensitivity

            return photon_flux
    
class SequentialCalibratePhotons(CalibratePhotons):
    def __init__(self, data_array_movie):

        # We call the parent class
        super().__init__(data_array_movie)

        # This is part of the sequential object data clean up. 
        self.data_array_movie = np.maximum(0, self.data_array_movie.astype(np.int32, copy=False))

        self.photon_sensitivity = None
        self.dark_signal = None
        self.fitted_pixels = None
        self.fitted_pixels_var = None
        self.fitted_pixels_mean = None
        self.fitted_model = None

    def _longest_run(self, bool_array: np.ndarray) -> slice:
        """
        Find the longest contiguous segment of True values inside bool_array.
        Args:
            bool_array: 1d boolean array.
        Returns:
            Slice with start and stop for the longest contiguous block of True values.
        """
        step = np.diff(np.int8(bool_array), prepend=0, append=0)
        on = np.where(step == 1)[0]
        off = np.where(step == -1)[0]
        i = np.argmax(off - on)
        return slice(on[i], off[i])


    def get_photon_sensitivity_parameters(self, count_weight_gamma: float=0.2) -> dict:
        """Calculate photon sensitivity

        Args:
            count_weight_gamma: 0.00001=weigh each intensity level equally, 
                1.0=weigh each intensity in proportion to pixel counts.

        Returns:
            A list with the photon gain and offset for each group of pixels.
        """

        intensity = (self.data_array_movie[:-1, :, :] + self.data_array_movie[1:, :, :] + 1) // 2
        difference = self.data_array_movie[:-1, :, :].astype(np.float32) - self.data_array_movie[1:, :, :]

        select = intensity > 0
        intensity = intensity[select]
        difference = difference[select]

        counts = np.bincount(intensity.flatten())
        bins = self._longest_run(counts > 0.01 * counts.mean())  # consider only bins with at least 1% of mean counts 
        bins = slice(max(bins.stop * 3 // 100, bins.start), bins.stop)
        assert (
            bins.stop - bins.start > 100
        ), f"Bins.start: {bins.start}, Bins.stop: {bins.stop}  The image does not have a sufficient range of intensities to compute the noise transfer function."

        counts = counts[bins]
        idx = (intensity >= bins.start) & (intensity < bins.stop)
        variance = (
            np.bincount(
                intensity[idx] - bins.start,
                weights=(difference[idx] ** 2) / 2,
            )
            / counts
        )
        model = Regressor()
        model.fit(np.c_[bins], variance, counts ** count_weight_gamma)
        sensitivity = model.coef_[0]
        zero_level = - model.intercept_ / model.coef_[0]

        self.photon_sensitivity = sensitivity
        self.dark_signal = zero_level
        self.fitted_pixels_var = variance
        self.fitted_pixels_mean = np.c_[bins]
        self.fitted_model = model

        return [self.photon_sensitivity, self.dark_signal]

    def plot_poisson_curve(self):
        """Obtain a plot showing Poisson characteristics of the signal.

        Returns:
            A figure.
        """
        if self.fitted_pixels_mean is None:
            raise ValueError("You need to compute the photon gain parameters first.")
        else:
            fig = plt.figure()

            plt.scatter(self.fitted_pixels_mean, self.fitted_pixels_var, s=1)
            mean_range = np.linspace(self.fitted_pixels_mean.min(), self.fitted_pixels_mean.max(), num=200)

            plt.plot(mean_range, (mean_range - self.dark_signal)* self.photon_sensitivity, 'r')
            plt.grid(True)
            plt.xlabel('intensity')
            plt.ylabel('variance')

            return fig

class RasterCalibratePhotons(CalibratePhotons):
    def __init__(self, data_array_movie):

        # We call the parent class
        super().__init__(data_array_movie)

        self.photon_sensitivity = None
        self.dark_signal = None
        self.group_images = None
        self.image_gain = None
        self.image_offset = None
        self.fitted_pixels = None
        self.fitted_pixels_var = None
        self.fitted_pixels_mean = None


    def plot_assignment_image(self):
        """Plot the assignment image. This shows the pixels that are assigned to each group."""
        """Each group correspond to pixels associated with a different photon gain and offset."""

        if self.group_images is None:
            raise ValueError("You need to compute the photon gain parameters first.")
        
        fig = plt.figure()

        size_x_subplots = 2
        size_y_subplots = len(self.group_images) // size_x_subplots + 1

        for local_index, local_image in enumerate(self.group_images):
            plt.subplot(size_y_subplots, size_x_subplots, local_index + 1)
            plt.title(f"Group {local_index}")
            plt.imshow(local_image, cmap="gray", interpolation='none')
            plt.axis("off")
        
        return fig  
    
    def plot_photon_sensitivity_image(self):
        """Plot the photon gain and offset images. These are the images that show the photon gain and offset for each pixel."""

        if self.image_gain is None:
            raise ValueError("You need to compute the photon gain parameters first.")
        
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(self.image_gain, cmap="gray", interpolation='none')
        plt.colorbar()
        plt.axis("off")
        plt.title("Photon Gain")

        plt.subplot(1, 2, 2)
        plt.imshow(self.image_offset, cmap="gray", interpolation='none')
        plt.colorbar()
        plt.axis("off")
        plt.title("Photon Offset")

        return fig
    
    def plot_poisson_curve(self):
        """Obtain a plot showing Poisson characteristics of the signal.

        Returns:
            A figure.
        """
        if self.fitted_pixels_mean is None:
            raise ValueError("You need to compute the photon gain parameters first.")
        else:
            fig = plt.figure()
        
            h, xedges, yedges = np.histogram2d(
                self.fitted_pixels_var, self.fitted_pixels_mean, bins=(200, 200)
            )
            extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]

            plt.imshow(h, origin="lower", extent=extent, aspect="auto", cmap="Blues")
            plt.colorbar()
            plt.xlabel("Mean")
            plt.ylabel("Variance")
            plt.xlim(self.fitted_pixels_mean.min(), self.fitted_pixels_mean.max())
            plt.ylim(self.fitted_pixels_var.min(), self.fitted_pixels_var.max())

            mean_range = np.linspace(self.fitted_pixels_mean.min(), self.fitted_pixels_mean.max(), num=200)

            for index, local_gain in enumerate(self.photon_sensitivity):
                local_offset = self.dark_signal[index]

                plt.tight_layout()
                plt.plot(
                    mean_range,
                    local_gain * (mean_range - local_offset),
                    'r', 
                    label=f"Line {index}", 
                )
            
            plt.legend()

            return fig

    def get_photon_sensitivity_parameters(self, max_pixel_range=2**15, n_groups=1, perc_min=3, perc_max=90):
        """Photon Gain.

        Extract the photon gain parameters from the data. This is useful for understanding the
        characteristics of the data and for calibrating the data.
        We assume there are n_groups of pixels to fit, each with their own photon gain and offset.
        When dealing with raster scanning microscopes, there can be multiple groups of pixels with
        different characteristics.

        Args:
            n_groups:  The number of groups of pixels to fit. This is useful for separating pixels
            with different characteristics. An optimization will be performed to match the 
            data with n_groups lines.
            max_pixel_range:  This is the maximum pixel value that is considered to be saturated.
            This is useful for removing saturated pixels from the analysis.
            perc_min, perc_max:  Min and max values between 0-100 used in filtering based on percentile.
            This is useful for removing pixels that deviate from Poisson statistics, for example if their 
            mean fluctuates too much due to other sources of signal in the data. 

        Returns:
            A list with the photon gain and offset for each group of pixels.
        """

        # Remove saturated pixels
        idxs_not_saturated = np.where(self.data_array_movie.max(axis=0).flatten() < max_pixel_range)

        _var = self.data_array_movie.var(axis=0).flatten()[idxs_not_saturated]
        _mean = self.get_mean_image().flatten()[idxs_not_saturated]

        # Remove pixels that deviate from Poisson stats
        _var_scale = np.percentile(_var, [perc_min, perc_max])
        _mean_scale = np.percentile(_mean, [perc_min, perc_max])

        # Remove outliers
        _var_bool = np.logical_and(_var > _var_scale[0], _var < _var_scale[1])
        _mean_bool = np.logical_and(_mean > _mean_scale[0], _mean < _mean_scale[1])
        _no_outliers = np.logical_and(_var_bool, _mean_bool)

        _var_filt = _var[_no_outliers]
        _mean_filt = _mean[_no_outliers]
        
        self.fitted_pixels = idxs_not_saturated[0][_no_outliers]
        self.fitted_pixels_var = _var_filt
        self.fitted_pixels_mean = _mean_filt

        if n_groups == 1:
            nb_attempts = 1
            print("Fitting a single line, a single attempt will be made, since this is a convex problem.")
                  
        else:
            nb_attempts = 5
            print(f"Fitting {n_groups} lines, {nb_attempts} attempts will be made, since this is a non-convex problem.")

        found_fits = self.fit_xlines(_var_filt, _mean_filt, n_groups, nb_attempts)

        photon_sensitivity_list = []
        dark_signal_list = []

        for i in range(found_fits.shape[0]):
            slope = found_fits[i, 0] 
            offset = found_fits[i, 1]

            photon_sensitivity_list.append(slope)
            dark_signal_list.append(-offset/slope)

        self.photon_sensitivity = np.array(photon_sensitivity_list)
        self.dark_signal = np.array(dark_signal_list)

        return [self.photon_sensitivity, self.dark_signal]

    # Define the regression model (line equation)
    def linear_model(self, params_flat, x, num_lines):
        """Linear model for fitting multiple lines."""

        # Reshape the flattened parameters
        params = params_flat.reshape((-1, 2))
        
        # We add as many rows to x as there are parameters
        X = np.vstack([x for i in range(num_lines)])
        
        # Calculate the predicted y-values for all lines
        y_predicted = np.multiply(params[:, 0], X.T) + np.multiply(params[:, 1], np.ones(X.T.shape))

        return y_predicted

    # Define the sum of squared differences as the objective function
    def objective(self, params_flat, x, y_observed, num_lines):
        """Objective function for fitting multiple lines."""

        all_predicted_y = self.linear_model(params_flat, x, num_lines).T
        
        Y = np.vstack([y_observed for i in range(num_lines)])
        # Calculate the sum of squared differences for all lines
        local_error = (Y - all_predicted_y)**2
        
        # get the best line for each point
        best_fit = np.min(local_error, axis=0)
        # get the total error across all points
        total_error = np.sum(best_fit)
        
        return total_error

    def fit_xlines(self, variance_array, mean_array, n_groups, nb_attempts):
        """Fit multiple lines to the data. This is useful for separating pixels with different characteristics."""

        _mat = np.vstack([mean_array, np.ones(len(mean_array))]).T

        # We first fit a single line to reference the data for all the lines
        slope, offset = np.linalg.lstsq(_mat, variance_array, rcond=None)[0]

        # define the model
        training_data = np.column_stack((mean_array, (variance_array-offset)/slope))
        
        current_best_error = np.inf

        for iteration in np.arange(nb_attempts):
            initial_guesses_flat = np.array([1, 0]*n_groups)+0.1*(np.random.rand(2*n_groups)-0.5)
            local_result = minimize(self.objective, initial_guesses_flat, args=(training_data[:,0], training_data[:,1], n_groups), method='Powell')
            error = local_result.fun
            
            print(f"Attempt {iteration+1} - Error: {error}")
            if error<current_best_error:
                current_best_error = error
                result = local_result
            
        found_lines = result.x.reshape((-1, 2))

        # We convert back to original coordinate system
        found_lines[:,0] = found_lines[:,0]*slope
        found_lines[:,1] = found_lines[:,1]*slope + offset

        print(f"Found lines: {found_lines}")

        return found_lines

    def get_pixel_assignement_images(self):
        """Get the pixel assignment images. This is useful for understanding the pixels that are assigned to each group."""

        if self.photon_sensitivity is None:
            raise ValueError("You need to compute the photon gain parameters first.")
        
        if self.group_images is not None:
            return self.group_images, self.image_gain, self.image_offset

        image_project = self.get_mean_image()
        pixel_coords = self.fitted_pixels
        
        # We measure the fitting error for each pixel and for each line
        all_error = []
        for index, local_gain in enumerate(self.photon_sensitivity): 
            local_offset = self.dark_signal[index]   
            error = (self.fitted_pixels_var - local_gain * (self.fitted_pixels_mean - local_offset))**2
            all_error.append(error)
        all_error = np.array(all_error)

        # We assign each pixel to the line that minimizes the error
        closest = np.argmin(all_error, axis = 0)
        
        image_gain = np.nan*np.ones(image_project.shape).flatten()
        image_offset = np.nan*np.ones(image_project.shape).flatten()
        group_images = []
        for local_index, local_gain in enumerate(self.photon_sensitivity): 
            plt.subplot(2,2,local_index+1)
            selected_pixels = np.where(closest==local_index)[0]
            local_project_copy = np.zeros(image_project.shape).flatten()
            local_project_copy[pixel_coords[selected_pixels]]=1
            image_gain[pixel_coords[selected_pixels]]=local_gain
            image_offset[pixel_coords[selected_pixels]]=self.dark_signal[local_index]
            local_project_copy = local_project_copy.reshape(image_project.shape)
            group_images.append(local_project_copy)

        image_gain = image_gain.reshape(image_project.shape)
        image_offset = image_offset.reshape(image_project.shape)
        
        self.group_images = group_images
        self.image_gain = image_gain
        self.image_offset = image_offset

        return group_images, image_gain, image_offset