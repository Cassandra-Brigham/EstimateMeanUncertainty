
import numpy as np
import rioxarray as rio
import rasterio
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import rasterio
import geopandas as gpd
import math
import pandas as pd
from numba import njit, prange

def dropna(array):
    """
    Drops NaN values from a NumPy array.

    Parameters:
    -----------
    array : np.ndarray
        The input NumPy array from which to drop NaN values.

    Returns:
    --------
    np.ndarray
        A new array with NaN values removed.
    """
    return array[~np.isnan(array)]

class RasterDataHandler:
    """
    A class used to loading vertical differencing raster data, 
    subtracting a vertical systematic error from from the raster, and randomly sampling the raster data for further analysis.

    Attributes:
    -----------
    raster_path : str
        The file path to the raster data.
    rioxarray_obj : rio.xarray_raster.Raster
        The rioxarray object holding the raster data.
    data_array : numpy.ndarray
        The loaded raster data as a numpy array, excluding NaN values.

    Methods:
    --------
    load_raster(masked=True)
        Loads the raster data from the given path, optionally applying a mask to exclude NaN values.
    subtract_value_from_raster(output_raster_path, value_to_subtract)
        Subtracts a given value from the raster data and saves the result to a new file.
    sample_raster(samples_per_sq_km, max_samples)
        Samples the raster data based on a given density and maximum number of samples, returning the sample values and their coordinates.
    """
    def __init__(self, raster_path, unit, resolution):
        """
        Parameters:
        -----------
        raster_path : str
            The file path to the raster data.
        """
        self.raster_path = raster_path
        self.unit = unit
        self.resolution = resolution
        self.rioxarray_obj = None
        self.data_array = None
        self.samples = None
        self.coords = None

    def load_raster(self, masked=True):
        """
        Loads the raster data from the specified path, applying a mask to exclude NaN values if requested.

        Parameters:
        -----------
        masked : bool, optional
            If True, NaN values in the raster data will be masked (default is True).
        """
        self.rioxarray_obj = rio.open_rasterio(self.raster_path, masked=masked)
        self.data_array = self.rioxarray_obj.data[~np.isnan(self.rioxarray_obj.data)].flatten()

    def subtract_value_from_raster(self, output_raster_path, value_to_subtract):
        """
        Subtracts a specified value from the raster data and saves the resulting raster to a new file.

        Parameters:
        -----------
        output_raster_path : str
            The file path where the modified raster will be saved.
        value_to_subtract : float
            The value to be subtracted from each pixel in the raster data.
        """
        with rasterio.open(self.raster_path) as src:
            data = src.read()
            nodata = src.nodata
            mask = data != nodata if nodata is not None else np.ones(data.shape, dtype=bool)
            data = data.astype(float)
            data[mask] -= value_to_subtract

            out_meta = src.meta.copy()
            out_meta.update({'dtype': 'float32', 'nodata': nodata})

            with rasterio.open(output_raster_path, 'w', **out_meta) as dst:
                dst.write(data)
                
    def plot_raster(self):
        fig,ax=plt.subplots(figsize=(10, 6))
        self.rioxarray_obj.plot(cmap="bwr_r", vmin=np.percentile(self.data_array,2),vmax=np.percentile(self.data_array, 98), ax=ax)
        ax.set_title("Vertical differencing results corrected for vertical bias (m)")
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        ax.ticklabel_format(style="plain")
        ax.set_aspect('equal')
        return fig

    def sample_raster(self, area_side, samples_per_area, max_samples):
        """
        Samples the raster data based on a given density (samples per square kilometer) 
        and a maximum number of samples to limit analysis time.
        Returns the sampled values and their corresponding coordinates.
        """
        with rasterio.open(self.raster_path) as src:
            data = src.read(1)
            nodata = src.nodata
            
            # Mask to find valid data
            valid_data_mask = data != nodata
            
            # Calculate area per pixel in square kilometers
            cell_size = src.res[0]
            cell_area_sq_km = (cell_size ** 2) / (area_side ** 2)
            
            # Find valid data points and calculate total samples needed
            valid_data_indices = np.where(valid_data_mask)
            valid_data_count = valid_data_indices[0].size
            
            total_samples = min(int(cell_area_sq_km * samples_per_area * valid_data_count), max_samples)
            
            if total_samples > valid_data_count:
                raise ValueError("Requested samples exceed valid data points.")
            
            # Randomly select valid data points
            chosen_indices = np.random.choice(valid_data_count, size=total_samples, replace=False)
            rows = valid_data_indices[0][chosen_indices]
            cols = valid_data_indices[1][chosen_indices]
            
            # Get sampled data values
            samples = data[rows, cols]
            
            # Compute coordinates all at once for efficiency
            x_coords, y_coords = src.xy(rows, cols)
            coords = np.vstack([x_coords, y_coords]).T
            
            mask = ~np.isnan(samples)
            filtered_samples = samples[mask]
            filtered_coords = coords[mask]
            
            # Store samples and coordinates
            self.samples = filtered_samples
            self.coords = filtered_coords
    
    

class StatisticalAnalysis:
    """
    A class to perform statistical analysis on data, including plotting data statistics and estimating the uncertainty of the median value of the data using bootstrap resampling with subsamples of the data.

    Methods:
    --------
    plot_data_stats(data):
        Plots the histogram of the data and overlays statistical measures like mean, median, and mode.
    bootstrap_uncertainty_subsample(data, n_bootstrap=1000, subsample_proportion=0.1):
        Estimates the uncertainty of the median value of the data using bootstrap resampling.
    """
    def __init__(self, raster_data_handler):
        """
        Parameters:
        -----------
        raster_data_handler : RasterDataHandler
        An instance of RasterDataHandler to manage raster data operations.
        """
        self.raster_data_handler = raster_data_handler

    
    def plot_data_stats(self):
        data = self.raster_data_handler.data_array
        mean = np.mean(data)
        median = np.median(data)
        mode_result = stats.mode(data, nan_policy='omit')  # This returns a ModeResult object.
        mode_val = mode_result.mode  # This might be an array or a single value.
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        p1 = np.percentile(data, 1)
        p99 = np.percentile(data, 99)
        minimum = np.min(data)
        maximum = np.max(data)
        
        filtered_data = data[(data >= p1) & (data <= p99)]
        
        # Ensure mode_val is iterable. If it's a single value, make it a list.
        if not isinstance(mode_val, np.ndarray):
            mode_val = [mode_val]  # Make it a list so it's iterable.

        fig, ax = plt.subplots()
        
        ax.hist(filtered_data, bins=60, density=False, alpha=0.6, color='g')
        ax.axvline(mean, color='r', linestyle='dashed', linewidth=1, label='Mean')
        ax.axvline(median, color='b', linestyle='dashed', linewidth=1, label='Median')
        for m in mode_val:  # Plot each mode
            ax.axvline(m, color='purple', linestyle='dashed', linewidth=1, label='Mode' if m == mode_val[0] else "_nolegend_")
        
        # Preparing the mode string for multi-modal data
        mode_str = ", ".join([f'{m:.3f}' for m in mode_val])
        
        textstr = '\n'.join((
            f'Mean: {mean:.3f}',
            f'Median: {median:.3f}',
            f'Mode(s): {mode_str}',
            f'Minimum: {minimum:.3f}',
            f'Maximum: {maximum:.3f}',
            f'1st Quartile: {q1:.3f}',
            f'3rd Quartile: {q3:.3f}'))
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        ax.set_xlabel(f'Vertical Difference ({self.raster_data_handler.unit})')
        ax.set_ylabel('Count')
        ax.set_title('Histogram of differencing results with exploratory statistics')
        ax.legend()
        plt.tight_layout()

        return fig

    
    def bootstrap_uncertainty_subsample(self, n_bootstrap=1000, subsample_proportion=0.1):
        """
        Estimates the uncertainty of the median value of the data using bootstrap resampling. This method randomly
        samples subsets of the data, calculates their medians, and then computes the standard deviation of these
        medians as a measure of uncertainty.

        Parameters:
        -----------
        n_bootstrap : int, optional
            The number of bootstrap samples to generate (default is 1000).
        subsample_proportion : float, optional
            The proportion of the data to include in each subsample (default is 0.1).

        Returns:
        --------
        uncertainty : float
            The standard deviation of the bootstrap medians, representing the uncertainty of the median value.
        """
        
        subsample_size = int(subsample_proportion * len(self.raster_data_handler.data_array))
        bootstrap_medians = np.zeros(n_bootstrap)
        for i in range(n_bootstrap):
            sample = np.random.choice(self.raster_data_handler.data_array, size=subsample_size, replace=True)
            bootstrap_medians[i] = np.median(sample)
        return np.std(bootstrap_medians)

class VariogramAnalysis:
    """
    A class to perform variogram analysis on raster data. It calculates mean variograms, 
    fits spherical models to the variogram data, and plots the results.

    Attributes:
    -----------
    raster_data_handler : RasterDataHandler
        An instance of RasterDataHandler to manage raster data operations.
    mean_variogram : numpy.ndarray
        The mean variogram calculated from multiple runs.
    lags : numpy.ndarray
        The distances (lags) at which the variogram is calculated.
    mean_count : numpy.ndarray
        The mean count of data pairs used for each lag distance.
    err_variogram : numpy.ndarray
        The standard deviation of the variogram values across multiple runs.
    fitted_variogram : numpy.ndarray
        The variogram values fitted using a model.
    rmse : float
        The root mean square error of the fitted model.
    sills : list
        The sill values of the fitted spherical model components.
    ranges : list
        The range values of the fitted spherical model components.
    err_param : numpy.ndarray
        The standard error of the parameters of the fitted model.
    initial_guess : list
        The initial guess parameters for the fitting process.

    Methods:
    --------
    calculate_mean_variogram(n_bins, n_runs)
        Calculates the mean variogram and its error from multiple runs.
    fit_3_spherical_models_no_nugget()
        Fits three spherical models without a nugget effect to the mean variogram.
    plot_3_spherical_models_no_nugget()
        Plots the mean variogram, its error, and the fitted model.
    """
    def __init__(self, raster_data_handler):
        """
        Initializes the VariogramAnalysis class with a RasterDataHandler instance.

        Parameters:
        -----------
        raster_data_handler : RasterDataHandler
            The RasterDataHandler instance to manage raster data operations.
        """
        self.raster_data_handler = raster_data_handler
        self.mean_variogram = None
        self.lags = None
        self.mean_count = None
        self.err_variogram = None
        self.fitted_variogram = None
        self.rmse = None
        self.sills = None
        self.ranges = None
        self.err_param = None
        self.err_ranges = None
        self.err_sills = None
        self.initial_guess = None
        self.ranges_min = None
        self.ranges_max = None
        self.sills_min = None
        self.sills_max = None
        self.min_unique_samples = None
        self.best_aic = None
        self.best_params = None
        self.best_model_config = None
        self.best_nugget = None


    def numba_variogram(self, area_side, samples_per_area, max_samples, bin_width,max_lag_multiplier):
        
        self.raster_data_handler.sample_raster(area_side, samples_per_area, max_samples)
        
        def pairwise_calc_python(coords, values):
            M = coords.shape[0]
            N = coords.shape[1]
            
            distances = np.empty((M, M), dtype=np.float64)
            abs_differences = np.empty((M, M), dtype=np.float64)
            
            for i in range(M):
                for j in range(M):
                    d = 0.0
                    abs_differences[i, j] = abs(values[i] - values[j])
                    for k in range(N):
                        tmp = coords[i, k] - coords[j, k]
                        d += tmp * tmp
                    distances[i, j] = np.sqrt(d)
            
            return distances, abs_differences


        # Pairwise distances and differences
        pairwise_calc_numba = njit(pairwise_calc_python)
        pairwise_distances, pairwise_abs_diff = pairwise_calc_numba(self.raster_data_handler.coords, self.raster_data_handler.samples)
        
        # Max distance and lag for binning
        max_distance = np.max(pairwise_distances)
        if max_lag_multiplier == "median":
            max_lag = np.median(pairwise_distances)
        elif max_lag_multiplier == "max":
            max_lag = max_distance
        else:
            max_lag = max_distance*max_lag_multiplier
        min_distance = 0.0

        n_bins = int((max_lag - min_distance) / bin_width) + 1

        # Create bins for distances
        bins = np.linspace(min_distance, n_bins * bin_width, n_bins + 1)

        # Digitize distances to assign them to bins
        bin_indices = np.digitize(pairwise_distances, bins) - 1  # Subtract 1 to correct bin indexing
        
        # Preallocate arrays for tracking differences and counts per bin
        max_pairs_per_bin = (self.raster_data_handler.coords.shape[0] ** 2) // 2  # Approximate upper bound
        differences = np.zeros((n_bins, max_pairs_per_bin), dtype=np.float64)  # 2D array to store differences
        bin_counts = np.zeros(n_bins, dtype=np.int32)  # Track counts per bin
        variogram_matheron = np.zeros(n_bins, dtype=np.float64)  # Store variogram values
        
        @njit
        def populate_bins_unique_numba(bin_indices, pairwise_abs_diff, differences, bin_counts, num_points, n_bins):
            # Populate bins without using a list of arrays
            for i in range(num_points):
                for j in range(i + 1, num_points):  # Only calculate upper triangle to avoid redundancy
                    bin_idx = bin_indices[i, j]
                    if bin_idx < n_bins:
                        diff_value = pairwise_abs_diff[i, j]
                        count = bin_counts[bin_idx]
                        differences[bin_idx, count] = diff_value
                        bin_counts[bin_idx] += 1
            
            return bin_counts, differences

        bin_counts, differences = populate_bins_unique_numba(bin_indices, pairwise_abs_diff, differences, bin_counts, self.raster_data_handler.coords.shape[0], n_bins)
        
        # Trim excess zeros from 'differences'
        trimmed_differences = [differences[b][:bin_counts[b]] for b in range(n_bins)]
        
        #@njit
        def matheron(x):
            # prevent ZeroDivisionError
            if x.size == 0:
                return np.nan

            return (1. / (2 * x.size)) * np.sum(np.power(x, 2))
        
        # Calculate the variogram for each bin
        for i in prange(n_bins):
            #variogram_dowd[i] = 2.198 * np.nanmedian(trimmed_differences[i]) ** 2 / 2
            variogram_matheron[i] = matheron(trimmed_differences[i])
        
        for i in prange(n_bins):
            variogram_matheron[i] = (1. / (2 * trimmed_differences[i].size)) * np.sum(np.power(trimmed_differences[i], 2))
            

        return bin_counts, variogram_matheron, n_bins, min_distance, max_distance
    
    def calculate_mean_variogram_numba(self, area_side, samples_per_area, max_samples, bin_width, max_n_bins, n_runs,max_lag_multiplier=1/3):
        """
        Calculates the mean variogram and its error from multiple sampling and calculation runs.

        Parameters:
        -----------
        n_bins : int
            The number of bins to divide the distance range into for variogram calculation.
        n_runs : int
            The number of runs to perform for sampling and variogram calculation.
        """
        # Initialize DataFrames and arrays
        all_variograms = pd.DataFrame(np.nan, index=range(n_runs), columns=range(max_n_bins))
        counts = pd.DataFrame(np.nan, index=range(n_runs), columns=range(max_n_bins))
    
        # Initialize arrays to store bin information
        all_n_bins = np.zeros(n_runs)
        min_distances = np.zeros(n_runs)
        max_distances = np.zeros(n_runs)
        
        for run in range(n_runs):
            # Calculate variogram for all runs
            count, variogram, n_bins, min_distance, max_distance = self.numba_variogram(area_side, samples_per_area, max_samples, bin_width,max_lag_multiplier)
            
            # Store the results
            all_variograms.loc[run, :variogram.size-1] = pd.Series(variogram).loc[:variogram.size-1]
            counts.loc[run, :count.size-1] = pd.Series(count).loc[:count.size-1]
            all_n_bins[run] = n_bins
            min_distances[run] = min_distance
            max_distances[run] = max_distance
        
        # Calculate mean and std dev of variograms across all runs
        mean_variogram = dropna(np.nanmean(all_variograms, axis=0))
        mean_count = dropna(np.nanmean(counts, axis=0))
        err_variogram = dropna(np.nanstd(all_variograms, axis=0))
        lags = np.linspace(bin_width / 2, (len(mean_variogram))*bin_width - bin_width / 2, len(mean_variogram))
        
        
        self.mean_variogram = mean_variogram
        self.err_variogram = err_variogram
        self.mean_count = mean_count
        self.lags = lags
    
    def fit_best_spherical_model(self):
        """
        Fits nested spherical models with 1, 2, and 3 components, both with and without a nugget effect.
        Selects and saves the parameters of the model with the lowest AIC.
        Additionally, returns RMSE, errors for ranges and sills, and other model details.
        """
        def spherical_model(h, *params):
            n = len(params) // 2
            C = params[:n]
            a = params[n:]
            model = np.zeros_like(h)
            for i in range(n):
                mask = h <= a[i]
                model[mask] += C[i] * (3 * h[mask] / (2 * a[i]) - (h[mask] ** 3) / (2 * a[i] ** 3))
                model[~mask] += C[i]
            return model

        def spherical_model_with_nugget(h, nugget, *params):
            return nugget + spherical_model(h, *params)
        
        lags = self.lags
        mean_variogram = self.mean_variogram
        sigma_filtered = self.err_variogram
        sigma_filtered[sigma_filtered==0]= np.finfo(float).eps

        # Define bounds and initial guesses for 1, 2, and 3 models
        model_configs = [
            {'components': 1, 'nugget': False},
            {'components': 1, 'nugget': True},
            {'components': 2, 'nugget': False},
            {'components': 2, 'nugget': True},
            {'components': 3, 'nugget': False},
            {'components': 3, 'nugget': True},
        ]
        
        best_aic = np.inf
        best_params = None
        best_model = None
        best_rmse = None
        best_err_param = None
        best_rmse_filtered = None
        best_err_ranges = None
        best_err_sills = None
        best_ranges_min = None
        best_ranges_max = None
        best_sills_min = None
        best_sills_max = None
        best_nugget = None

        for config in model_configs:
            n = config['components']
            nugget = config['nugget']
            
            lower_bounds = [0] * n + [1]* n + int(nugget)*[0]
            upper_bounds = [np.max(mean_variogram)] * n + [np.max(lags)] * n + ([np.max(mean_variogram)] if nugget else [])
            
            # Generate initial guess
            max_semivariance = np.max(mean_variogram)
            half_max_lag = np.max(lags) / 2
            C = [max_semivariance / n] * n
            a = [half_max_lag * (i + 1) / n for i in range(n)]
            p0 = C + a + ([max_semivariance / 10] if nugget else [])
            
            bounds = (lower_bounds, upper_bounds)
            
            model_func = spherical_model_with_nugget if nugget else spherical_model
            
            try:
                popt, pcov = curve_fit(model_func, lags, mean_variogram, p0=p0, bounds=bounds, sigma=sigma_filtered, maxfev=10000)
                
                fitted_variogram = model_func(self.lags, *popt)
                residuals = self.mean_variogram - fitted_variogram
                n_data_points = len(self.mean_variogram)
                residual_sum_of_squares = np.sum(residuals**2)
                sigma_squared = np.var(residuals)
                if sigma_squared <= 0: # Avoid taking the log of zero or negative variance
                    sigma_squared = np.finfo(float).eps  # Use a small value instead
                log_likelihood = -0.5 * n_data_points * np.log(2 * np.pi * sigma_squared) - residual_sum_of_squares / (2 * sigma_squared)
                k = len(popt)
                aic = 2 * k - 2 * log_likelihood
                err_param = np.sqrt(np.diag(pcov))
                
                rmse_filtered = np.sqrt(np.mean((mean_variogram - model_func(lags, *popt))**2))
                rmse = np.sqrt(np.mean((self.mean_variogram - fitted_variogram)**2))
                
                if nugget:
                    ranges = popt[n:n+n] 
                    sills = popt[0:n]
                    err_ranges = err_param[n:n+n]
                    err_sills = err_param[0:n]
                    nugget_value = popt[-1]
                else:
                    ranges = popt[n:n+n] 
                    sills = popt[0:n]
                    err_ranges = err_param[n:n+n]
                    err_sills = err_param[0:n]
                    nugget_value = None
                
                ranges_min = [ranges[a] - err_ranges[a] for a in range(0, len(ranges))]
                ranges_max = [ranges[a] + err_ranges[a] for a in range(0, len(ranges))]
                sills_min = [sills[a] - err_sills[a] for a in range(0, len(sills))]
                sills_max = [sills[a] + err_sills[a] for a in range(0, len(sills))]

                if aic < best_aic:
                    best_aic = aic
                    best_params = popt
                    best_model = config
                    best_rmse = rmse
                    best_err_param = err_param
                    best_rmse_filtered = rmse_filtered
                    best_err_ranges = err_ranges
                    best_err_sills = err_sills
                    best_ranges_min = ranges_min
                    best_ranges_max = ranges_max
                    best_sills_min = sills_min
                    best_sills_max = sills_max
                    best_nugget = nugget_value
                    best_ranges = ranges
                    best_sills = sills
            except RuntimeError:
                continue
        
        if best_model is not None:
            self.best_aic = best_aic
            self.best_params = best_params
            self.best_model_config = best_model
            self.best_rmse = best_rmse
            self.best_err_param = best_err_param
            self.best_rmse_filtered = best_rmse_filtered
            self.ranges = best_ranges
            self.err_ranges = best_err_ranges
            self.sills = best_sills
            self.err_sills = best_err_sills
            self.ranges_min = best_ranges_min
            self.ranges_max = best_ranges_max
            self.sills_min = best_sills_min
            self.sills_max = best_sills_max
            self.best_nugget = best_nugget

            self.fitted_variogram = spherical_model_with_nugget(self.lags, *best_params) if best_model['nugget'] else spherical_model(self.lags, *best_params)
            
        
    def plot_best_spherical_model(self):
        """
        Plots the mean variogram with error bars, the fitted spherical model, and annotations for ranges and sills.
        This visualization helps in understanding the spatial structure of the data, including both AIC and RMSE metrics.
        """
        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]}, figsize=(10, 8))
        
         # Histogram of semivariance counts
        ax[0].bar(self.lags[:-1], self.mean_count[:-1], width=np.diff(self.lags)[0] * 0.9, color='orange', alpha=0.5)
        ax[0].set_ylabel('Mean Count')

        # Plot mean variogram with error bars indicating spread over the 10 runs
        ax[1].errorbar(self.lags[:-1], self.mean_variogram[:-1], yerr=self.err_variogram[:-1], fmt='o-', color='blue', label='Mean Variogram with Spread')

        # Plot fitted model
        if self.fitted_variogram is not None:
            ax[1].plot(self.lags[:-1], self.fitted_variogram[:-1], 'r-', label='Fitted Model')

        colors = ['red', 'green', 'blue']
        if self.ranges is not None and self.err_ranges is not None:
            for i, (range_val, error_val) in enumerate(zip(self.ranges, self.err_ranges)):
                color = colors[i]
                # Bold line at range value
                ax[1].axvline(x=range_val, color=color, linestyle='--', lw=1, label=f'Range {i+1}')
                # Lighter lines at +/- 1 std error
                ax[1].fill_betweenx([0, np.max(self.mean_variogram)], range_val - error_val, range_val + error_val, color=color, alpha=0.2)

        # Plot the sills with error bounds if available
        #if self.sills is not None and self.err_sills is not None:
        #    for i, (sill_val, error_val) in enumerate(zip(self.sills, self.err_sills)):
        #        ax[1].axhline(y=sill_val, color=colors[i], linestyle='-', lw=1, label=f'Sill {i+1}')
        #        ax[1].fill_betweenx([sill_val - error_val, sill_val + error_val], 0, np.max(self.lags), color=colors[i], alpha=0.1)

        # Check if a nugget effect was used
        if self.best_nugget is not None:
            ax[1].axhline(y=self.best_nugget, color='black', linestyle='--', lw=1, label='Nugget Effect')

        ax[1].set_xlabel('Lag Distance')
        ax[1].set_ylabel('Semivariance')
        ax[1].legend()

        # Show both AIC and RMSE in the title
       #aic_str = f'AIC: {self.best_aic:.4f}' if self.best_aic is not None else "AIC: N/A"
        rmse_str = f'RMSE: {self.best_rmse:.4f}' if self.best_rmse is not None else "RMSE: N/A"
        #ax[1].set_title(f'{aic_str}, {rmse_str}')
        ax[1].set_title(f'{rmse_str}')

        plt.setp(ax[0].get_xticklabels(), visible=False)
        plt.tight_layout()

        return fig

class UncertaintyCalculation:
    """
    A class designed to calculate various types of uncertainty associated with spatial data,
    particularly focusing on the uncertainty derived from variogram analysis.

    Attributes:
    -----------
    variogram_analysis : VariogramAnalysis
        An instance of VariogramAnalysis containing variogram results and fitted model parameters.
    mean_random_uncorrelated : float
        The mean random uncertainty not correlated to any spatial structure.
    mean_random_correlated_1 : float
        The mean random uncertainty correlated to the first spherical model.
    mean_random_correlated_2 : float
        The mean random uncertainty correlated to the second spherical model.
    mean_random_correlated_3 : float
        The mean random uncertainty correlated to the third spherical model.
    total_mean_uncertainty : float
        The total mean uncertainty calculated as a combination of uncorrelated and correlated uncertainties.

    Methods:
    --------
    calc_mean_random_uncorrelated()
        Calculates the mean random uncertainty not correlated to any spatial structure.
    calc_mean_random_correlated(dem_resolution=1)
        Calculates the mean random uncertainty correlated with the spatial structure defined by the variogram's spherical models.
    calc_total_mean_uncertainty()
        Calculates the total mean uncertainty, combining both correlated and uncorrelated uncertainties.
    """

    def __init__(self, variogram_analysis):
        """
        Initializes the UncertaintyCalculation class with a VariogramAnalysis instance.

        Parameters:
        -----------
        variogram_analysis : VariogramAnalysis
            An instance of VariogramAnalysis for accessing variogram results and model parameters.
        """
        self.variogram_analysis = variogram_analysis
        # Initialize all uncertainty attributes to None.
        self.mean_random_uncorrelated = None
        self.mean_random_correlated_1 = None
        self.mean_random_correlated_2 = None
        self.mean_random_correlated_3 = None
        self.mean_random_correlated_1_min = None
        self.mean_random_correlated_2_min = None
        self.mean_random_correlated_3_min = None
        self.mean_random_correlated_1_max = None
        self.mean_random_correlated_2_max = None
        self.mean_random_correlated_3_max = None
        self.total_mean_uncertainty = None
        self.total_mean_uncertainty_min = None
        self.total_mean_uncertainty_max = None
        self.area=None

    def calc_mean_random_uncorrelated(self):
        """
        Calculates the mean random uncertainty not correlated to any spatial structure,
        based on the standard deviation of the data and the number of data points.
        """
        data = self.variogram_analysis.raster_data_handler.data_array

        def calculate_rms(values):
            """Calculate the Root Mean Square (RMS) of an array of numbers."""
            # Step 1: Square all the numbers
            squared_values = [x**2 for x in values]
            
            # Step 2: Calculate the mean of the squares
            mean_of_squares = sum(squared_values) / len(values)
            
            # Step 3: Take the square root of the mean
            rms = math.sqrt(mean_of_squares)
    
            return rms
        
        rms = calculate_rms(data)

        self.mean_random_uncorrelated = rms / np.sqrt(len(data))

    def calc_mean_random_correlated(self):
        """
        Calculates the mean random uncertainties correlated with the spatial structure defined by
        the variogram's spherical models.

        """
        dem_resolution = self.variogram_analysis.raster_data_handler.resolution
        data = self.variogram_analysis.raster_data_handler.data_array
        # Calculate correlated uncertainties for each spherical model component.
        self.mean_random_correlated_1 = (np.sqrt(2 * self.variogram_analysis.sills[0]) / np.sqrt(len(data))) * np.sqrt((np.pi * np.square(self.variogram_analysis.ranges[0])) / (5 * np.square(dem_resolution)))
        self.mean_random_correlated_2 = (np.sqrt(2 * self.variogram_analysis.sills[1]) / np.sqrt(len(data))) * np.sqrt((np.pi * np.square(self.variogram_analysis.ranges[1])) / (5 * np.square(dem_resolution)))
        self.mean_random_correlated_3 = (np.sqrt(2 * self.variogram_analysis.sills[2]) / np.sqrt(len(data))) * np.sqrt((np.pi * np.square(self.variogram_analysis.ranges[2])) / (5 * np.square(dem_resolution)))
        self.area=dem_resolution*len(data)

    def calc_mean_random_correlated_min(self):
        """
        Calculates the minimum mean random uncertainties correlated with the spatial structure defined by
        the variogram's spherical models, as defined by the optimal range and sill values minus 1 std error.
        """
        dem_resolution = self.variogram_analysis.raster_data_handler.resolution
        data = self.variogram_analysis.raster_data_handler.data_array
        # Calculate correlated uncertainties for each spherical model component.
        if self.variogram_analysis.sills_min[0]>0:
            self.mean_random_correlated_1_min = (np.sqrt(2 * self.variogram_analysis.sills_min[0]) / np.sqrt(len(data))) * np.sqrt((np.pi * np.square(self.variogram_analysis.ranges_min[0])) / (5 * np.square(dem_resolution)))
        else:
            self.mean_random_correlated_1_min = 0

        if self.variogram_analysis.sills_min[1]>0:
            self.mean_random_correlated_2_min = (np.sqrt(2 * self.variogram_analysis.sills_min[1]) / np.sqrt(len(data))) * np.sqrt((np.pi * np.square(self.variogram_analysis.ranges_min[1])) / (5 * np.square(dem_resolution)))
        else:
            self.mean_random_correlated_2_min = 0
        if self.variogram_analysis.sills_min[2]>0:
            self.mean_random_correlated_3_min = (np.sqrt(2 * self.variogram_analysis.sills_min[2]) / np.sqrt(len(data))) * np.sqrt((np.pi * np.square(self.variogram_analysis.ranges_min[2])) / (5 * np.square(dem_resolution)))
        else:
            self.mean_random_correlated_3_min = 0

    def calc_mean_random_correlated_max(self):
        """
        Calculates the maximum mean random uncertainties correlated with the spatial structure defined by
        the variogram's spherical models, as defined by the optimal range and sill values plus 1 std error.
        """
        dem_resolution = self.variogram_analysis.raster_data_handler.resolution
        data = self.variogram_analysis.raster_data_handler.data_array
        # Calculate correlated uncertainties for each spherical model component.
        self.mean_random_correlated_1_max = (np.sqrt(2 * self.variogram_analysis.sills_max[0]) / np.sqrt(len(data))) * np.sqrt((np.pi * np.square(self.variogram_analysis.ranges_max[0])) / (5 * np.square(dem_resolution)))
        self.mean_random_correlated_2_max = (np.sqrt(2 * self.variogram_analysis.sills_max[1]) / np.sqrt(len(data))) * np.sqrt((np.pi * np.square(self.variogram_analysis.ranges_max[1])) / (5 * np.square(dem_resolution)))
        self.mean_random_correlated_3_max = (np.sqrt(2 * self.variogram_analysis.sills_max[2]) / np.sqrt(len(data))) * np.sqrt((np.pi * np.square(self.variogram_analysis.ranges_max[2])) / (5 * np.square(dem_resolution)))
        
    
   
    def calc_total_mean_uncertainty(self):
        """
        Calculates the total mean uncertainty by adding in quadrature the uncertainties (both correlated and uncorrelated).
        """
        self.total_mean_uncertainty = np.sqrt(np.square(self.mean_random_uncorrelated) + np.square(self.mean_random_correlated_1) + np.square(self.mean_random_correlated_2) + np.square(self.mean_random_correlated_3))
    
    def calc_total_mean_uncertainty_min(self):
        """
        Calculates the minimum total mean uncertainty by adding in quadrature the uncertainties (both minimum correlated and uncorrelated).
        """
        self.total_mean_uncertainty_min = np.sqrt(np.square(self.mean_random_uncorrelated) + np.square(self.mean_random_correlated_1_min) + np.square(self.mean_random_correlated_2_min) + np.square(self.mean_random_correlated_3_min))

    def calc_total_mean_uncertainty_max(self):
        """
        Calculates the maximum total mean uncertainty by adding in quadrature the uncertainties (both maximum correlated and uncorrelated).
        """
        self.total_mean_uncertainty_max = np.sqrt(np.square(self.mean_random_uncorrelated) + np.square(self.mean_random_correlated_1_max) + np.square(self.mean_random_correlated_2_max) + np.square(self.mean_random_correlated_3_max))

