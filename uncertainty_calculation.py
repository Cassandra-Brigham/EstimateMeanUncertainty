
import numpy as np
import rioxarray as rio
import rasterio
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
import rasterio
import geopandas as gpd
import math
import variogram_tools
import pandas as pd

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
        Samples the raster data based on a given density (samples per square kilometer) and a maximum number of samples to limit analysis time.
        Returns the sampled values and their corresponding coordinates.

        Parameters:
        -----------
        samples_per_sq_km : int
            The desired number of samples per square kilometer.
        max_samples : int
            The maximum number of samples to be taken.

        Returns:
        --------
        samples : numpy.ndarray
            An array of sampled vertical differencing values (in m) from the raster data.
        coords : numpy.ndarray
            The coordinates corresponding to each sampled value.
        """
        with rasterio.open(self.raster_path) as src:
            data, nodata = src.read(1), src.nodata
            valid_data_mask = data != nodata
            cell_size, cell_area_sq_km = src.res[0], (src.res[0] ** 2) / (area_side ** 2)
            valid_data_points = np.sum(valid_data_mask)
            total_samples = min(int(cell_area_sq_km * samples_per_area * valid_data_points), max_samples)

            if total_samples > valid_data_points:
                raise ValueError("Requested samples exceed valid data points.")
            
            chosen_indices = np.random.choice(np.where(valid_data_mask.ravel())[0], size=total_samples, replace=False)
            samples = data.ravel()[chosen_indices]
            coords = np.array([src.xy(index // data.shape[1], index % data.shape[1]) for index in chosen_indices])

            return samples, coords
    
    

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
        self.rmse_filtered = None
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
        
    def calculate_mean_variogram(self, sample_area_side, samples_per_area, max_samples, bin_width, max_n_bins, n_runs):
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
        all_unique_samples = pd.DataFrame(np.nan, index=range(n_runs), columns=range(max_n_bins))
    
        # Initialize arrays to store bin information
        all_n_bins = np.zeros(n_runs)
        min_distances = np.zeros(n_runs)
        max_distances = np.zeros(n_runs)
        
        for run in range(n_runs):
            # Sample raster
            samples, coords = self.raster_data_handler.sample_raster(sample_area_side,samples_per_area,max_samples)
            # Calculate bin widths and min and max distances
            n_bins, min_distance, max_distance = variogram_tools.calculate_n_bins(coords, width=bin_width)
            # Calculate variogram for this run
            variogram, count, unique_samples = variogram_tools.dowd_estimator_cy(coords.astype(np.float64), samples.astype(np.float64), n_bins, bin_width, min_distance, max_distance)
            
            # Store the results
            all_variograms.loc[run, :variogram.size-1] = pd.Series(variogram).loc[:variogram.size-1]
            counts.loc[run, :count.size-1] = pd.Series(count).loc[:count.size-1]
            all_unique_samples.loc[run, :unique_samples.size-1] = pd.Series(unique_samples).loc[:unique_samples.size-1]
            all_n_bins[run] = n_bins
            min_distances[run] = min_distance
            max_distances[run] = max_distance
        
        max_all_bins = int(np.nanmax(all_n_bins))
        
        # Calculate mean and std dev of variograms across all runs
        mean_variogram = dropna(np.nanmean(all_variograms, axis=0))
        mean_count = dropna(np.nanmean(counts, axis=0))
        err_variogram = dropna(np.nanstd(all_variograms, axis=0))
        unique_samples = dropna(np.nanmin(all_unique_samples, axis=0))
        lags = np.linspace(bin_width / 2, (len(mean_variogram))*bin_width - bin_width / 2, len(mean_variogram))
        
        # Drop bins with fewer than 150 unique samples
        unique_samples_drop = unique_samples[unique_samples>150]
        mean_count_drop = mean_count[unique_samples>150]
        mean_variogram_drop = mean_variogram[unique_samples>150]
        err_variogram_drop = err_variogram[unique_samples>150]
        lags_drop = lags[unique_samples>150]
        
        self.mean_variogram = mean_variogram_drop
        self.err_variogram = err_variogram_drop
        self.mean_count = mean_count_drop
        self.min_unique_samples = unique_samples_drop
        
        # Assuming bin_widths are similar across runs, use the first one to calculate bin lags
        self.lags = lags_drop
       #self.lags = np.linspace(bin_width / 2, (max_lag-1) - bin_width / 2, (max_lag-1)*bin_width)


    def fit_3_spherical_models_no_nugget(self):
        """
        Fits three spherical models without a nugget effect to the mean variogram data.
        This method sets the fitted model parameters, RMSE, ranges, and sills attributes.
        """
        
        condition1 = self.err_variogram < np.max(self.mean_variogram)/2 
        condition2 = self.lags <= np.percentile(self.lags,95)
        condition=condition1 & condition2
        lags = self.lags[condition]
        mean_variogram = self.mean_variogram[condition]

        # Define bounds
        lower_bounds = [0, 0, 0, 0, 0, 0]
        upper_bounds = [np.max(mean_variogram), np.max(lags), np.max(mean_variogram), np.max(lags), np.max(mean_variogram), np.max(lags)]
        bounds = (lower_bounds, upper_bounds)

        #Initial guess
        def initial_guess_func(mean_variogram,lags,preset_lags = None):
            max_semivariance = np.max(mean_variogram) 
            C1 = 0.20 * max_semivariance
            C2 = 0.20 * max_semivariance + C1
            C3 = max_semivariance - (C1 + C2) 

            if preset_lags is None:
                half_max_lag = np.max(lags)/2
                a1 = 0.25*half_max_lag
                a2 = 0.5*half_max_lag
                a3 = 0.75*half_max_lag
            else:
                a1, a2, a3 = preset_lags

            initial_guess = [C1, a1, C2, a2, C3, a3]
            return initial_guess
            
        intial_guess = initial_guess_func(mean_variogram,lags)
        self.initial_guess = intial_guess

        # Define spherical model with no nugget
        def spherical_model_no_nugget(h, C1, a1, C2, a2, C3, a3):
            model = np.zeros_like(h)
            for C, a in zip([C1, C2, C3], [a1, a2, a3]):
                mask = h <= a
                model[mask] += C * (3*h[mask]/(2*a) - (h[mask]**3)/(2*a**3))
                model[~mask] += C
            return model
        
        sigma_filtered = self.err_variogram[condition]
        sigma_non_zero = sigma_filtered[sigma_filtered > 0]
        sigma_non_zero = sigma_non_zero[sigma_non_zero != np.nan]

        sigma_filtered[sigma_filtered < np.min(sigma_non_zero)] = np.min(sigma_non_zero)

        # Perform the curve fitting with bounds
        popt, pcov = curve_fit(spherical_model_no_nugget, lags, mean_variogram, sigma=sigma_filtered,absolute_sigma=False, p0=self.initial_guess, bounds=bounds, maxfev=10000)

        # Calculate the standard deviations (errors) of the optimized parameters
        self.err_param = np.sqrt(np.diag(pcov))

        # Calculate RMSE using only the filtered data
        self.fitted_variogram = spherical_model_no_nugget(self.lags, *popt)  
        self.rmse_filtered = np.sqrt(np.mean((mean_variogram - spherical_model_no_nugget(lags, *popt))**2))
        self.rmse = np.sqrt(np.mean((self.mean_variogram - self.fitted_variogram)**2))
        self.ranges = [popt[i] for i in range(1, len(popt), 2)]
        self.err_ranges = [self.err_param[i] for i in range(1, len(self.err_param), 2)]
        self.sills = [popt[i] for i in range(0, len(popt), 2)]
        self.err_sills = [self.err_param[i] for i in range(0, len(self.err_param), 2)]

        # Get the values of the ranges and sills of the +/- 1 std error of the optimal curve parmeters
        self.ranges_min = [self.ranges[a]-self.err_ranges[a] for a in range(0,len(self.ranges))]
        self.ranges_max = [self.ranges[a]+self.err_ranges[a] for a in range(0,len(self.ranges))]

        self.sills_min = [self.sills[a]-self.err_sills[a] for a in range(0,len(self.sills))]
        self.sills_max = [self.sills[a]+self.err_sills[a] for a in range(0,len(self.sills))]
            
    def plot_3_spherical_models_no_nugget(self):
        """
        Plots the mean variogram with error bars, the fitted spherical model, and annotations for ranges and sills.
        This visualization helps in understanding the spatial structure of the data.
        """
        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]}, figsize=(10, 8))

        # Histogram of semivariance counts
        ax[0].bar(self.lags[:-1], self.min_unique_samples[:-1], width=np.diff(self.lags)[0]*0.9, color='orange', alpha=0.5)
        #ax[0].set_ylabel('Mean Number\nof Pairs')
        ax[0].set_ylabel('Min Number\nof Unique Samples')

        # Plot mean variogram with error bars indicating spread over the 10 runs
        ax[1].errorbar(self.lags[:-1], self.mean_variogram[:-1], yerr=self.err_variogram[:-1], fmt='o-', color='blue', label='Mean Variogram with Spread')

        # Plot fitted model 
        ax[1].plot(self.lags[:-1], self.fitted_variogram[:-1], 'r-', label='Fitted Model')

        colors = ['red', 'green', 'blue']
        for i, (range, error) in enumerate(zip(self.ranges, self.err_ranges)):
            color = colors[i]
            # Bold line at range value
            ax[1].axvline(x=range, color=color,linestyle='--', lw=1, label=f'Range {i+1}')
            # Lighter lines at +/- 1 std error
            #ax[1].axvline(x=range - error, color=color, linestyle='--', lw=1)
            #ax[1].axvline(x=range + error, color=color, linestyle='--', lw=1)
            # Translucent area (using fill_betweenx to fill vertically)
            #ax[1].fill_betweenx([0, self.sills[i]], range - error, range + error, color=color, alpha=0.2)
       
        ax[1].set_xlabel('Lag Distance'), ax[1].set_ylabel('Semivariance')
        ax[1].legend()
        ax[1].set_title(f'RMSE: {self.rmse:.4f}')
        
        plt.setp(ax[0].get_xticklabels(), visible=False)
        plt.tight_layout()#, plt.show()

        return fig
    
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

        for config in model_configs:
            n = config['components']
            nugget = config['nugget']
            
            lower_bounds = [0] * (2 * n + int(nugget))
            upper_bounds = [np.max(mean_variogram)] * n + [np.max(lags)] * n + ([np.max(mean_variogram)] if nugget else [])
            
            # Generate initial guess
            max_semivariance = np.max(mean_variogram)
            half_max_lag = np.max(lags) / 2
            C = [max_semivariance / n] * n
            a = [half_max_lag * (i + 1) / n for i in range(n)]
            p0 = C + a + ([max_semivariance / 10] if nugget else [])
            
            bounds = (lower_bounds, upper_bounds)
            
            # Choose the model function
            if nugget:
                model_func = spherical_model_with_nugget
            else:
                model_func = spherical_model
            
            # Fit the model
            try:
                popt, pcov = curve_fit(model_func, lags, mean_variogram, p0=p0, bounds=bounds, sigma=sigma_filtered, maxfev=10000)
                
                # Calculate the fitted values
                fitted_variogram = model_func(self.lags, *popt)
                
                # Calculate the residuals
                residuals = self.mean_variogram - fitted_variogram
                
                # Calculate the log-likelihood
                n_data_points = len(self.mean_variogram)
                residual_sum_of_squares = np.sum(residuals**2)
                sigma_squared = np.var(residuals)
                log_likelihood = -0.5 * n_data_points * np.log(2 * np.pi * sigma_squared) - residual_sum_of_squares / (2 * sigma_squared)
                
                # Calculate the AIC
                k = len(popt)
                aic = 2 * k - 2 * log_likelihood
                
                # Calculate the standard deviations (errors) of the optimized parameters
                err_param = np.sqrt(np.diag(pcov))

                # Calculate RMSE using only the filtered data
                rmse_filtered = np.sqrt(np.mean((mean_variogram - model_func(lags, *popt))**2))
                rmse = np.sqrt(np.mean((self.mean_variogram - fitted_variogram)**2))
                
                # Extract ranges and sills
                if nugget:
                    ranges = [popt[i] for i in range(2, len(popt), 2)]
                    sills = [popt[i] for i in range(1, len(popt), 2)]
                    err_ranges = [err_param[i] for i in range(2, len(err_param), 2)]
                    err_sills = [err_param[i] for i in range(1, len(err_param), 2)]
                else:
                    ranges = [popt[i] for i in range(1, len(popt), 2)]
                    sills = [popt[i] for i in range(0, len(popt), 2)]
                    err_ranges = [err_param[i] for i in range(1, len(err_param), 2)]
                    err_sills = [err_param[i] for i in range(0, len(err_param), 2)]

                # Get the values of the ranges and sills of the +/- 1 std error of the optimal curve parameters
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

            except RuntimeError:
                continue  # Skip this configuration if fitting fails
        
        if best_model is not None:
            self.best_aic = best_aic
            self.best_params = best_params
            self.best_model_config = best_model
            self.best_rmse = best_rmse
            self.best_err_param = best_err_param
            self.best_rmse_filtered = best_rmse_filtered
            self.ranges = [best_params[i] for i in range(1, len(best_params), 2)]
            self.err_ranges = best_err_ranges
            self.sills = [best_params[i] for i in range(0, len(best_params), 2)]
            self.err_sills = best_err_sills
            self.ranges_min = best_ranges_min
            self.ranges_max = best_ranges_max
            self.sills_min = best_sills_min
            self.sills_max = best_sills_max

            self.fitted_variogram = spherical_model_with_nugget(self.lags, *best_params) if best_model['nugget'] else spherical_model(self.lags, *best_params)
        
    def plot_best_spherical_model(self):
        """
        Plots the mean variogram with error bars, the fitted spherical model, and annotations for ranges and sills.
        This visualization helps in understanding the spatial structure of the data, including both AIC and RMSE metrics.
        """
        fig, ax = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1, 3]}, figsize=(10, 8))

        # Histogram of semivariance counts
        ax[0].bar(self.lags[:-1], self.min_unique_samples[:-1], width=np.diff(self.lags)[0] * 0.9, color='orange', alpha=0.5)
        ax[0].set_ylabel('Min Number\nof Unique Samples')

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
        if self.sills is not None and self.err_sills is not None:
            for i, (sill_val, error_val) in enumerate(zip(self.sills, self.err_sills)):
                ax[1].axhline(y=sill_val, color=colors[i], linestyle='-', lw=1, label=f'Sill {i+1}')
                ax[1].fill_betweenx([sill_val - error_val, sill_val + error_val], 0, np.max(self.lags), color=colors[i], alpha=0.1)

        # Check if a nugget effect was used
        if self.best_model_config.get('nugget', False) and self.best_nugget is not None:
            ax[1].axhline(y=self.best_nugget, color='black', linestyle='--', lw=1, label='Nugget Effect')

        ax[1].set_xlabel('Lag Distance')
        ax[1].set_ylabel('Semivariance')
        ax[1].legend()

        # Show both AIC and RMSE in the title
        aic_str = f'AIC: {self.best_aic:.4f}' if self.best_aic is not None else "AIC: N/A"
        rmse_str = f'RMSE: {self.rmse:.4f}' if self.rmse is not None else "RMSE: N/A"
        ax[1].set_title(f'{aic_str}, {rmse_str}')

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

