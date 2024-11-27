
import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import rasterio
from rasterio import plot as rio_plot
import rioxarray as rio
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
    A class used for loading vertical differencing raster data, 
    subtracting a vertical systematic error from the raster, and randomly sampling the raster data for further analysis.

    Attributes:
    -----------
    raster_path : str
        The file path to the raster data.
    unit : str
        The unit of measurement for the raster data.
    resolution : float
        The resolution of the raster data.
    rioxarray_obj : rioxarray.DataArray
        The rioxarray object holding the raster data.
    data_array : numpy.ndarray
        The loaded raster data as a numpy array, excluding NaN values.
    transformed_values : numpy.ndarray
        The transformed values after applying a normal score transform.
    sorted_data : numpy.ndarray
        The sorted data used for normal score transformation.
    normal_scores : numpy.ndarray
        The normal scores corresponding to the sorted data.
    samples : numpy.ndarray
        The sampled values from the raster data.
    coords : numpy.ndarray
        The coordinates of the sampled values.
    normal_transform_raster_path : str
        The file path to the raster with normal score transformed values.
    normal_transform_rioxarray_obj : rioxarray.DataArray
        The rioxarray object holding the normal score transformed raster data.
    normal_transform_data_array : numpy.ndarray
        The normal score transformed raster data as a numpy array, excluding NaN values.

    Methods:
    --------
    load_raster(masked=True)
        Loads the raster data from the given path, optionally applying a mask to exclude NaN values.
    subtract_value_from_raster(output_raster_path, value_to_subtract)
        Subtracts a given value from the raster data and saves the result to a new file.
    normal_score_transform(data)
        Transforms the data to normal scores using rank-based empirical CDF.
    map_transformed_values_to_raster(output_raster_normal_transform)
        Maps the transformed values back to a rioxarray DataArray of the original size and resolution.
    transform_data(output_normal_transform_raster_path)
        Transforms the data using the normal score transform.
    plot_raster(plot_title, normal_transform=True)
        Plots a raster image using the rioxarray object.
    sample_raster(area_side, samples_per_area, max_samples, normal_transform=True)
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
        self.transformed_values = None
        self.sorted_data = None
        self.normal_scores = None
        self.samples = None
        self.coords = None
        self.normal_transform_raster_path = None
        self.normal_transform_rioxarray_obj = None
        self.normal_transform_data_array = None

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
    
    @staticmethod
    def normal_score_transform(data):
        """
        Transform the data to normal scores using rank-based empirical CDF.
        """
        sorted_data = np.sort(data)
        ranks = np.argsort(data)
        n = len(sorted_data)
        normal_scores = norm.ppf((np.arange(1, n + 1) - 0.5) / n)
        normal_score_data = np.zeros_like(data)
        normal_score_data[ranks] = normal_scores
        return normal_score_data, sorted_data, normal_scores
    
    def map_transformed_values_to_raster(self,output_raster_normal_transform):
        """
        Maps the transformed values back to a rioxarray DataArray of the original size and resolution.
        
        Returns:
        --------
        rioxarray_obj : xarray.DataArray
            The rioxarray object with the transformed values mapped to the original raster's size and resolution.
        """
        with rasterio.open(self.raster_path) as src:
            data = src.read(1)  # Read the first band
            nodata = src.nodata
            mask = data != nodata if nodata is not None else np.ones(data.shape, dtype=bool)
            
            # Ensure that the number of transformed values matches the number of True values in the mask
            num_valid_values = np.count_nonzero(mask)
            if len(self.transformed_values) != num_valid_values:
                raise ValueError(f"Mismatch in the number of valid pixels: mask has {num_valid_values} valid values, but transformed data has {len(self.transformed_values)} values.")
            
            # Create an output array filled with nodata values
            output_data = np.full(data.shape, nodata, dtype='float32')
            
            # Place the transformed values into the output array where the mask is True
            output_data[mask] = self.transformed_values
            
            # Copy the metadata of the original raster
            out_meta = src.meta.copy()
            out_meta.update({'dtype': 'float32', 'nodata': nodata})
            
            # Save the new raster
            with rasterio.open(output_raster_normal_transform, 'w', **out_meta) as dst: 
                dst.write(output_data, 1)
    
    def transform_data(self, output_normal_transform_raster_path):
        """

        Transforms the data using the normal score transform.

        """   
        self.transformed_values, self.sorted_data, self.normal_scores = self.normal_score_transform(self.rioxarray_obj.data.flatten())
        self.normal_transform_raster_path = output_normal_transform_raster_path
        self.map_transformed_values_to_raster(self.normal_transform_raster_path)
        self.normal_transform_rioxarray_obj = rio.open_rasterio(self.normal_transform_raster_path, masked=True)
        self.normal_transform_data_array = self.normal_transform_rioxarray_obj.data[~np.isnan(self.normal_transform_rioxarray_obj.data)].flatten()
        
    def plot_raster(self, plot_title, normal_transform=True):
        """
        Plots a raster image using the rioxarray object.
        Parameters:
        plot_title (str): The title of the plot.
        normal_transform (bool): If True, applies a normal score transformation to the rioxarray object before plotting. Defaults to True.
        Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
        """
        if normal_transform:
            rio_data = self.normal_transform_rioxarray_obj
            title = plot_title+" - Normal Score Transformed"
        else:
            rio_data = self.rioxarray_obj
            title = plot_title
            
        fig,ax=plt.subplots(figsize=(10, 6))
        rio_data.plot(cmap="bwr_r", ax=ax, robust = True)
        ax.set_title(title, pad=30)
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')
        ax.ticklabel_format(style="plain")
        ax.set_aspect('equal')
        return fig

    def sample_raster(self, area_side, samples_per_area, max_samples, normal_transform=True):
        """
        Samples the raster data based on a given density (samples per square kilometer) 
        and a maximum number of samples to limit analysis time.
        Returns the sampled values and their corresponding coordinates.
        """
        if normal_transform:
            raster_path = self.normal_transform_raster_path
        else:
            raster_path = self.raster_path
        with rasterio.open(raster_path) as src:
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

    Attributes:
    -----------
    raster_data_handler : RasterDataHandler
        An instance of RasterDataHandler to manage raster data operations.

    Methods:
    --------
    plot_data_stats(normal_transform=True, filtered=True)
        Plots the histogram of the raster data with exploratory statistics.
    bootstrap_uncertainty_subsample(n_bootstrap=1000, subsample_proportion=0.1)
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

    
    def plot_data_stats(self, normal_transform = True, filtered = True):
        """
        Plots the histogram of the raster data with exploratory statistics.
        Parameters:
        -----------
        normal_transform : bool, optional
            If True, use the normal transformed data array. If False, use the original data array.
            Default is True.
        filtered : bool, optional
            If True, filter the data to exclude outliers (1st and 99th percentiles) for better visualization.
            If False, use the unfiltered data. Default is True.
        Returns:
        --------
        fig : matplotlib.figure.Figure
            The matplotlib figure object containing the histogram and statistics.
        Notes:
        ------
        - The function calculates and displays the mean, median, mode(s), minimum, maximum, 
            1st quartile, and 3rd quartile of the data.
        - The mode(s) are displayed as vertical dashed lines on the histogram.
        - A text box with the calculated statistics is added to the plot.
        - The histogram is plotted with 60 bins and the data is optionally filtered to exclude 
            outliers for better visualization.
        """
        if normal_transform:
            data = self.raster_data_handler.normal_transform_data_array
        else:
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
        
        if filtered:
            # Filter the data to exclude outliers for better visualization
            data = data[(data >= p1) & (data <= p99)]
        else:
            data = data
        
        # Ensure mode_val is iterable. If it's a single value, make it a list.
        if not isinstance(mode_val, np.ndarray):
            mode_val = [mode_val]  # Make it a list so it's iterable.

        fig, ax = plt.subplots()
        
        # Plot the histogram of the data
        ax.hist(data, bins=60, density=False, alpha=0.6, color='g')
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
    cell_declustering_weights(cell_size, n_offsets)
        Calculate declustering weights using cell declustering.
    generate_correlated_pairs(num_pairs, rho)
        Generate correlated standard normal pairs using Monte Carlo simulation.
    back_transform_variogram(variogram_values, num_pairs=100000)
        Back-transform the variogram from normal scores to original units using Monte Carlo simulation.
    numba_variogram(area_side, samples_per_area, max_samples, bin_width, cell_size, n_offsets, max_lag_multiplier, normal_transform, weights)
        Calculate the variogram using Numba for performance optimization.
    calculate_mean_variogram_numba(area_side, samples_per_area, max_samples, bin_width, max_n_bins, n_runs, cell_size, n_offsets, max_lag_multiplier=1/3, normal_transform=True, weights=True)
        Calculate the mean variogram using numba for multiple runs.
    fit_best_spherical_model()
        Fits the best spherical model to the variogram data.
    plot_best_spherical_model()
        Plots the best spherical model for the variogram analysis.
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
        self.normal_scores = None
        self.declustering_weights = None
        self.untransformed_sills = None
        self.untransformed_err_sills = None
        self.back_transformed_variogram = None


    def cell_declustering_weights(self, cell_size, n_offsets):
        """
        Calculate declustering weights using cell declustering.
        
        Parameters:
        - coordinates: (n, 2) array of x, y coordinates.
        - cell_size: Size of each cell for declustering.
        - n_offsets: Number of random origins to average weights.
        
        Returns:
        - weights: Declustering weights for each data point.
        """
        coordinates = self.raster_data_handler.coords
        n_points = coordinates.shape[0]
        weights = np.zeros(n_points)
        
        # Iterate over a number of random grid origins to stabilize the weights
        for _ in range(n_offsets):
            # Randomly choose grid origin offsets
            x_offset = np.random.uniform(0, cell_size)
            y_offset = np.random.uniform(0, cell_size)
            
            # Compute cell indices for each point
            cell_indices_x = np.floor((coordinates[:, 0] - x_offset) / cell_size).astype(int)
            cell_indices_y = np.floor((coordinates[:, 1] - y_offset) / cell_size).astype(int)
            cell_indices = np.vstack((cell_indices_x, cell_indices_y)).T
            
            # Count points in each cell
            unique_cells, counts = np.unique(cell_indices, axis=0, return_counts=True)
            cell_counts_dict = {tuple(cell): count for cell, count in zip(unique_cells, counts)}
            
            # Assign weights inversely proportional to the count in each cell
            for i in range(n_points):
                cell = (cell_indices_x[i], cell_indices_y[i])
                weights[i] += 1.0 / cell_counts_dict[cell]
        
        # Average weights over all grid offsets
        weights /= n_offsets
        
        # Normalize weights to sum to 1
        weights /= np.sum(weights)
        
        self.declustering_weights = weights
        
    @staticmethod
    @njit(parallel=True)
    def generate_correlated_pairs(num_pairs, rho):    
        """
        Generate correlated standard normal pairs using Monte Carlo simulation. 
        This function is JIT-compiled with Numba to improve performance.
        """
        y_ls1 = np.random.randn(num_pairs)
        y_ls2 = np.random.randn(num_pairs)
        y_l1 = y_ls1
        y_l2 = y_ls1 * rho + y_ls2 * np.sqrt(1 - rho ** 2)
        return y_l1, y_l2
    
    def back_transform_variogram(self, variogram_values, num_pairs=100000):
        """
        Back-transform the variogram from normal scores to original units using Monte Carlo simulation.
        """
        normal_scores = self.raster_data_handler.normal_scores
        sorted_data = self.raster_data_handler.sorted_data
        
        # Precompute interpolation function for the back-transformation
        percentiles = np.linspace(0, 1, len(normal_scores))
        back_transform_func = interp1d(norm.cdf(normal_scores), sorted_data, bounds_error=False, fill_value=(sorted_data[0], sorted_data[-1]))
        back_transformed_variogram = []
        for gamma_Y in variogram_values:
            rho = 1 - gamma_Y
            # Step 1: Generate correlated pairs
            y_l1, y_l2 = self.generate_correlated_pairs(num_pairs, rho)
            
            # Step 2: Vectorized back-transformation
            z_l1 = back_transform_func(norm.cdf(y_l1))
            z_l2 = back_transform_func(norm.cdf(y_l2))
            
            # Step 3: Compute the variogram in original units
            gamma_Z_h = np.mean((z_l1 - z_l2) ** 2) / 2
            back_transformed_variogram.append(gamma_Z_h)
        
        return np.array(back_transformed_variogram)
    
    def numba_variogram(self, area_side, samples_per_area, max_samples, bin_width, cell_size, n_offsets, max_lag_multiplier, normal_transform, weights):
        """
        Calculate the variogram using Numba for performance optimization.
        Parameters:
        -----------
        area_side : float
            The side length of the area to sample.
        samples_per_area : int
            The number of samples to take per area.
        max_samples : int
            The maximum number of samples to take.
        bin_width : float
            The width of the bins for distance binning.
        cell_size : float
            The size of the cell for declustering.
        n_offsets : int
            The number of offsets for declustering.
        max_lag_multiplier : str or float
            The multiplier for the maximum lag distance. Can be "median", "max", or a float value.
        normal_transform : bool
            Whether to apply a normal transformation to the data.
        weights : bool
            Whether to apply declustering weights.
        Returns:
        --------
        bin_counts : numpy.ndarray
            The counts of pairs in each bin.
        variogram_matheron : numpy.ndarray
            The calculated variogram values for each bin.
        n_bins : int
            The number of bins used.
        min_distance : float
            The minimum distance considered.
        max_distance : float
            The maximum distance considered.
        """
        
        self.raster_data_handler.sample_raster(area_side, samples_per_area, max_samples, normal_transform)
        
        if weights:
            self.cell_declustering_weights(cell_size, n_offsets)
        else:
            self.declustering_weights = None
        
        
        @njit(parallel=True)
        def compute_pairwise_numba(coords, values):
            M = coords.shape[0]
            distances = np.empty((M, M), dtype=np.float64)
            abs_differences = np.empty((M, M), dtype=np.float64)
            
            for i in prange(M):
                for j in range(i, M):  # Only compute upper triangle
                    d = 0.0
                    for k in range(coords.shape[1]):  # Iterate over dimensions
                        tmp = coords[i, k] - coords[j, k]
                        d += tmp * tmp
                    dist = np.sqrt(d)
                    distances[i, j] = dist
                    distances[j, i] = dist  # Symmetric matrix
                    
                    diff = abs(values[i] - values[j])
                    abs_differences[i, j] = diff
                    abs_differences[j, i] = diff  # Symmetric matrix
            
            return distances, abs_differences
        
        pairwise_distances, pairwise_abs_diff = compute_pairwise_numba(self.raster_data_handler.coords, self.raster_data_handler.samples)

        
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
        
        @njit(parallel=True)
        def populate_bins_unique_numba(bin_indices, pairwise_abs_diff, differences, bin_counts, num_points, n_bins):
            """
            Populate bins with unique pairwise absolute differences using Numba for optimization.

            Parameters:
            -----------
            bin_indices : ndarray
                A 2D array of bin indices for each pair of points.
            pairwise_abs_diff : ndarray
                A 2D array of pairwise absolute differences.
            differences : ndarray
                A 2D array to store the differences for each bin.
            bin_counts : ndarray
                A 1D array to count the number of elements in each bin.
            num_points : int
                The number of points.
            n_bins : int
                The number of bins.

            Returns:
            --------
            tuple
                Updated bin_counts and differences arrays.
            """
            # Populate bins without using a list of arrays
            for i in prange(num_points):
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
        
        @njit(parallel=True, fastmath=True)
        def matheron(x):
            """
            Calculate the Matheron estimator for a given array.

            The Matheron estimator is used to estimate the variance of a sample.
            This function prevents a ZeroDivisionError by returning NaN if the input array is empty.

            Parameters:
            x (numpy.ndarray): Input array for which the Matheron estimator is calculated.

            Returns:
            float: The Matheron estimator value. Returns NaN if the input array is empty.
            """
            # prevent ZeroDivisionError
            if x.size == 0:
                return np.nan

            return (1. / (2 * x.size)) * np.sum(np.power(x, 2))
        
        # Calculate the variogram for each bin
        for i in prange(n_bins):
            #variogram_dowd[i] = 2.198 * np.nanmedian(trimmed_differences[i]) ** 2 / 2
            variogram_matheron[i] = matheron(trimmed_differences[i])
            
        return bin_counts, variogram_matheron, n_bins, min_distance, max_distance
    
    def calculate_mean_variogram_numba(self, area_side, samples_per_area, max_samples, bin_width, max_n_bins, n_runs, cell_size, n_offsets, max_lag_multiplier=1/3, normal_transform = True, weights = True):
        
        """
        Calculate the mean variogram using numba for multiple runs.
        Parameters:
        -----------
        area_side : float
            The side length of the area to be sampled.
        samples_per_area : int
            The number of samples to be taken per area.
        max_samples : int
            The maximum number of samples to be taken.
        bin_width : float
            The width of each bin for the variogram.
        max_n_bins : int
            The maximum number of bins for the variogram.
        n_runs : int
            The number of runs to perform for averaging.
        cell_size : float
            The size of each cell in the grid.
        n_offsets : int
            The number of offsets to use in the variogram calculation.
        max_lag_multiplier : float, optional
            The maximum lag distance as a fraction of the area side length (default is 1/3).
        normal_transform : bool, optional
            Whether to apply a normal transformation to the data (default is True).
        weights : bool, optional
            Whether to use weights in the variogram calculation (default is True).
        Returns:
        --------
        None
        Attributes:
        -----------
        mean_variogram : numpy.ndarray
            The mean variogram calculated across all runs.
        err_variogram : numpy.ndarray
            The standard deviation of the variogram across all runs.
        mean_count : numpy.ndarray
            The mean count of pairs in each bin across all runs.
        lags : numpy.ndarray
            The lag distances corresponding to the variogram bins.
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
            count, variogram, n_bins, min_distance, max_distance = self.numba_variogram(area_side, samples_per_area, max_samples, bin_width, cell_size, n_offsets, max_lag_multiplier, normal_transform, weights)
            
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
        Fits the best spherical model to the variogram data.
        This method evaluates multiple spherical model configurations with and without a nugget effect,
        and selects the best model based on the Akaike Information Criterion (AIC). The spherical model
        is commonly used in geostatistics to describe spatial correlation.
        The method iterates over different configurations, fits the models using non-linear least squares,
        and computes various metrics such as AIC, RMSE, and parameter uncertainties. The best model and
        its parameters are stored in the instance attributes.
        Attributes:
        self.lags (array-like): Array of lag distances.
        self.mean_variogram (array-like): Array of mean variogram values.
        self.err_variogram (array-like): Array of variogram errors.
        Updates the following instance attributes:
        self.best_aic (float): The best AIC value found.
        self.best_params (array-like): The best-fit parameters for the selected model.
        self.best_model_config (dict): Configuration of the best model.
        self.best_rmse (float): Root Mean Square Error of the best model.
        self.best_err_param (array-like): Uncertainties of the best-fit parameters.
        self.best_rmse_filtered (float): RMSE of the best model with filtered data.
        self.ranges (array-like): Range parameters of the best model.
        self.err_ranges (array-like): Uncertainties of the range parameters.
        self.sills (array-like): Sill parameters of the best model.
        self.err_sills (array-like): Uncertainties of the sill parameters.
        self.ranges_min (array-like): Minimum range values considering uncertainties.
        self.ranges_max (array-like): Maximum range values considering uncertainties.
        self.sills_min (array-like): Minimum sill values considering uncertainties.
        self.sills_max (array-like): Maximum sill values considering uncertainties.
        self.best_nugget (float or None): Nugget effect of the best model, if applicable.
        self.fitted_variogram (array-like): Fitted variogram values using the best model.
        Raises:
        RuntimeError: If the model fitting fails for a particular configuration.
        """
        
        def spherical_model(h, *params):
            """
            Computes the spherical model for given distances and parameters.

            The spherical model is commonly used in geostatistics to describe spatial 
            correlation. It is defined piecewise, with different formulas for distances 
            less than or equal to the range parameter and for distances greater than the 
            range parameter.

            Parameters:
            h (array-like): Array of distances at which to evaluate the model.
            *params: Variable length argument list containing the sill and range parameters.
                     The first half of the parameters are the sills (C), and the second half 
                     are the ranges (a). The number of sills and ranges should be equal.

            Returns:
            numpy.ndarray: Array of model values corresponding to the input distances.
            """
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
            """
            Computes the spherical model with a nugget effect.

            The spherical model is a type of variogram model used in geostatistics.
            This function adds a nugget effect to the spherical model.

            Parameters:
            h (array-like): Array of distances at which to evaluate the model.
            nugget (float): The nugget effect, representing the discontinuity at the origin.
            *params: Variable length argument list containing the sill and range parameters.
                     The first half of the parameters are the sills (C), and the second half 
                     are the ranges (a). The number of sills and ranges should be equal.

            Returns:
            numpy.ndarray: Array of model values corresponding to the input distances.
            """
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
                
                # Extract sill values
                if config['nugget']:
                    sills = popt[0:n]
                else:
                    sills = popt[0:n]
                    
                # Untransform sills to the original data space
                #untransformed_sills = np.array([self.inverse_transform_table.get(sill, sill) for sill in sills])
                err_sills = np.sqrt(np.diag(pcov)[0:n])
                
                # Untransform sill errors
                #untransformed_err_sills = np.array([self.inverse_transform_table.get(sill + err, err) - untransformed_sill
                #for sill, err, untransformed_sill in zip(sills, err_sills, untransformed_sills)]):
                    # Store the untransformed sills and their errors
                    #self.untransformed_sills = untransformed_sills
                    #self.untransformed_err_sills = untransformed_err_sills
                
                
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
        Plots the best spherical model for the variogram analysis.
        This function generates a two-panel plot:
        - The top panel displays a histogram of semivariance counts.
        - The bottom panel shows the mean variogram with error bars, the fitted model, range values with their errors, and the nugget effect if applicable.
        The plot includes:
        - A histogram of semivariance counts in the top panel.
        - The mean variogram with error bars indicating the spread over multiple runs.
        - The fitted variogram model if available.
        - Vertical lines indicating range values and shaded areas representing their errors.
        - A horizontal line indicating the nugget effect if used.
        - The RMSE value in the title of the bottom panel.
        Returns:
            fig (matplotlib.figure.Figure): The figure object containing the plot.
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
    mean_random_correlated_1_min : float
        The minimum mean random uncertainty correlated to the first spherical model.
    mean_random_correlated_2_min : float
        The minimum mean random uncertainty correlated to the second spherical model.
    mean_random_correlated_3_min : float
        The minimum mean random uncertainty correlated to the third spherical model.
    mean_random_correlated_1_max : float
        The maximum mean random uncertainty correlated to the first spherical model.
    mean_random_correlated_2_max : float
        The maximum mean random uncertainty correlated to the second spherical model.
    mean_random_correlated_3_max : float
        The maximum mean random uncertainty correlated to the third spherical model.
    total_mean_uncertainty : float
        The total mean uncertainty calculated as a combination of uncorrelated and correlated uncertainties.
    total_mean_uncertainty_min : float
        The minimum total mean uncertainty calculated as a combination of uncorrelated and correlated uncertainties.
    total_mean_uncertainty_max : float
        The maximum total mean uncertainty calculated as a combination of uncorrelated and correlated uncertainties.
    area : float
        The area associated with the uncertainty calculation.

    Methods:
    --------
    calc_mean_random_uncorrelated()
        Calculates the mean random uncorrelated uncertainty.
    calc_mean_random_correlated()
        Calculates the mean random correlated uncertainties for each spherical model component.
    calc_mean_random_correlated_min()
        Calculates the mean random correlated uncertainties for the minimum sills of the variogram analysis.
    calc_mean_random_correlated_max()
        Calculates the mean random correlated uncertainties for the maximum sills of the variogram analysis.
    calc_total_mean_uncertainty()
        Calculates the total mean uncertainty by adding in quadrature the uncertainties (both correlated and uncorrelated).
    calc_total_mean_uncertainty_min()
        Calculates the minimum total mean uncertainty by adding in quadrature the uncertainties (both minimum correlated and uncorrelated).
    calc_total_mean_uncertainty_max()
        Calculates the maximum total mean uncertainty by adding in quadrature the uncertainties (both maximum correlated and uncorrelated).
    """

    def __init__(self, variogram_analysis):
        """
        Initialize the UncertaintyCalculation class with a variogram analysis object.
        Parameters:
        variogram_analysis (object): An object containing variogram analysis data.
        Attributes:
        mean_random_uncorrelated (float or None): Mean random uncorrelated uncertainty.
        mean_random_correlated_1 (float or None): Mean random correlated uncertainty for the first correlation.
        mean_random_correlated_2 (float or None): Mean random correlated uncertainty for the second correlation.
        mean_random_correlated_3 (float or None): Mean random correlated uncertainty for the third correlation.
        mean_random_correlated_1_min (float or None): Minimum mean random correlated uncertainty for the first correlation.
        mean_random_correlated_2_min (float or None): Minimum mean random correlated uncertainty for the second correlation.
        mean_random_correlated_3_min (float or None): Minimum mean random correlated uncertainty for the third correlation.
        mean_random_correlated_1_max (float or None): Maximum mean random correlated uncertainty for the first correlation.
        mean_random_correlated_2_max (float or None): Maximum mean random correlated uncertainty for the second correlation.
        mean_random_correlated_3_max (float or None): Maximum mean random correlated uncertainty for the third correlation.
        total_mean_uncertainty (float or None): Total mean uncertainty.
        total_mean_uncertainty_min (float or None): Minimum total mean uncertainty.
        total_mean_uncertainty_max (float or None): Maximum total mean uncertainty.
        area (float or None): Area associated with the uncertainty calculation.
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
        Calculate the mean random uncorrelated uncertainty.
        This method calculates the mean random uncorrelated uncertainty by 
        performing the following steps:
        1. Retrieve the data array from the variogram analysis raster data handler.
        2. Define a nested function to calculate the Root Mean Square (RMS) of an array of numbers.
        3. Use the nested function to calculate the RMS of the data array.
        4. Compute the mean random uncorrelated uncertainty by dividing the RMS by the square root of the length of the data array.
        The result is stored in the `mean_random_uncorrelated` attribute of the instance.
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
        Calculate the mean random correlated uncertainties for each spherical model component.
        This method computes the correlated uncertainties based on the variogram analysis 
        parameters such as sills and ranges, and the resolution of the raster data. The 
        uncertainties are calculated for up to three spherical model components if available.
        Attributes:
        -----------
        mean_random_correlated_1 : float
            The mean random correlated uncertainty for the first spherical model component.
        mean_random_correlated_2 : float, optional
            The mean random correlated uncertainty for the second spherical model component, 
            if available.
        mean_random_correlated_3 : float, optional
            The mean random correlated uncertainty for the third spherical model component, 
            if available.
        area : float
            The total area covered by the raster data.
        """
        
        dem_resolution = self.variogram_analysis.raster_data_handler.resolution
        data = self.variogram_analysis.raster_data_handler.data_array
        # Calculate correlated uncertainties for each spherical model component.
        self.mean_random_correlated_1 = (np.sqrt(2 * self.variogram_analysis.sills[0]) / np.sqrt(len(data))) * np.sqrt((np.pi * np.square(self.variogram_analysis.ranges[0])) / (5 * np.square(dem_resolution)))
        if len(self.variogram_analysis.ranges)>1:
            self.mean_random_correlated_2 = (np.sqrt(2 * self.variogram_analysis.sills[1]) / np.sqrt(len(data))) * np.sqrt((np.pi * np.square(self.variogram_analysis.ranges[1])) / (5 * np.square(dem_resolution)))
            if len(self.variogram_analysis.ranges)>2:
                self.mean_random_correlated_3 = (np.sqrt(2 * self.variogram_analysis.sills[2]) / np.sqrt(len(data))) * np.sqrt((np.pi * np.square(self.variogram_analysis.ranges[2])) / (5 * np.square(dem_resolution)))    
        self.area=dem_resolution*len(data)

    def calc_mean_random_correlated_min(self):
        """
        Calculate the mean random correlated uncertainties for the minimum sills of the variogram analysis.
        This method calculates the mean random correlated uncertainties for each spherical model component
        based on the minimum sills and ranges from the variogram analysis. The calculation is performed
        only if the sill value is greater than zero. The results are stored in the attributes:
        `mean_random_correlated_1_min`, `mean_random_correlated_2_min`, and `mean_random_correlated_3_min`.
        Attributes:
            mean_random_correlated_1_min (float): Mean random correlated uncertainty for the first spherical model component.
            mean_random_correlated_2_min (float): Mean random correlated uncertainty for the second spherical model component.
            mean_random_correlated_3_min (float): Mean random correlated uncertainty for the third spherical model component.
        Returns:
            None
        """
        
        dem_resolution = self.variogram_analysis.raster_data_handler.resolution
        data = self.variogram_analysis.raster_data_handler.data_array
        # Calculate correlated uncertainties for each spherical model component.
        if self.variogram_analysis.sills_min[0]>0:
            self.mean_random_correlated_1_min = (np.sqrt(2 * self.variogram_analysis.sills_min[0]) / np.sqrt(len(data))) * np.sqrt((np.pi * np.square(self.variogram_analysis.ranges_min[0])) / (5 * np.square(dem_resolution)))
        else:
            self.mean_random_correlated_1_min = 0
        if len(self.variogram_analysis.ranges)>1:
            if self.variogram_analysis.sills_min[1]>0:
                self.mean_random_correlated_2_min = (np.sqrt(2 * self.variogram_analysis.sills_min[1]) / np.sqrt(len(data))) * np.sqrt((np.pi * np.square(self.variogram_analysis.ranges_min[1])) / (5 * np.square(dem_resolution)))
            else:
                self.mean_random_correlated_2_min = 0
            if len(self.variogram_analysis.ranges)>2:
                if self.variogram_analysis.sills_min[2]>0:
                    self.mean_random_correlated_3_min = (np.sqrt(2 * self.variogram_analysis.sills_min[2]) / np.sqrt(len(data))) * np.sqrt((np.pi * np.square(self.variogram_analysis.ranges_min[2])) / (5 * np.square(dem_resolution)))
                else:
                    self.mean_random_correlated_3_min = 0

    def calc_mean_random_correlated_max(self):
        """
        Calculate the mean random correlated uncertainties for each spherical model component.
        This method computes the mean random correlated uncertainties using the maximum sills and ranges
        from the variogram analysis. The calculation is performed for each spherical model component
        present in the variogram analysis.
        The formula used for the calculation is:
        mean_random_correlated = (sqrt(2 * sill) / sqrt(len(data))) * sqrt((pi * range^2) / (5 * dem_resolution^2))
        Attributes:
        -----------
        mean_random_correlated_1_max : float
            The mean random correlated uncertainty for the first spherical model component.
        mean_random_correlated_2_max : float, optional
            The mean random correlated uncertainty for the second spherical model component, if present.
        mean_random_correlated_3_max : float, optional
            The mean random correlated uncertainty for the third spherical model component, if present.
        Notes:
        ------
        - The method assumes that the variogram analysis has at least one spherical model component.
        - The method updates the instance attributes `mean_random_correlated_1_max`, `mean_random_correlated_2_max`,
          and `mean_random_correlated_3_max` based on the number of spherical model components.
        """
        
        dem_resolution = self.variogram_analysis.raster_data_handler.resolution
        data = self.variogram_analysis.raster_data_handler.data_array
        # Calculate correlated uncertainties for each spherical model component.
        self.mean_random_correlated_1_max = (np.sqrt(2 * self.variogram_analysis.sills_max[0]) / np.sqrt(len(data))) * np.sqrt((np.pi * np.square(self.variogram_analysis.ranges_max[0])) / (5 * np.square(dem_resolution)))
        if len(self.variogram_analysis.ranges)>1:
            self.mean_random_correlated_2_max = (np.sqrt(2 * self.variogram_analysis.sills_max[1]) / np.sqrt(len(data))) * np.sqrt((np.pi * np.square(self.variogram_analysis.ranges_max[1])) / (5 * np.square(dem_resolution)))
            if len(self.variogram_analysis.ranges)>2:
                self.mean_random_correlated_3_max = (np.sqrt(2 * self.variogram_analysis.sills_max[2]) / np.sqrt(len(data))) * np.sqrt((np.pi * np.square(self.variogram_analysis.ranges_max[2])) / (5 * np.square(dem_resolution)))    
             
    def calc_total_mean_uncertainty(self):
        """
        Calculates the total mean uncertainty by adding in quadrature the uncertainties (both correlated and uncorrelated).
        """
        if len(self.variogram_analysis.ranges) ==1:
            self.total_mean_uncertainty = np.sqrt(np.square(self.mean_random_uncorrelated) + np.square(self.mean_random_correlated_1))
        elif len(self.variogram_analysis.ranges) ==2:
            self.total_mean_uncertainty = np.sqrt(np.square(self.mean_random_uncorrelated) + np.square(self.mean_random_correlated_1) + np.square(self.mean_random_correlated_2))
        elif len(self.variogram_analysis.ranges) ==3:
            self.total_mean_uncertainty = np.sqrt(np.square(self.mean_random_uncorrelated) + np.square(self.mean_random_correlated_1) + np.square(self.mean_random_correlated_2) + np.square(self.mean_random_correlated_3))
    
    def calc_total_mean_uncertainty_min(self):
        """
        Calculates the minimum total mean uncertainty by adding in quadrature the uncertainties (both minimum correlated and uncorrelated).
        """
        if len(self.variogram_analysis.ranges) ==1:
            self.total_mean_uncertainty_min = np.sqrt(np.square(self.mean_random_uncorrelated) + np.square(self.mean_random_correlated_1_min))
        elif len(self.variogram_analysis.ranges) ==2:
            self.total_mean_uncertainty_min = np.sqrt(np.square(self.mean_random_uncorrelated) + np.square(self.mean_random_correlated_1_min) + np.square(self.mean_random_correlated_2_min))
        elif len(self.variogram_analysis.ranges) ==3:
            self.total_mean_uncertainty_min = np.sqrt(np.square(self.mean_random_uncorrelated) + np.square(self.mean_random_correlated_1_min) + np.square(self.mean_random_correlated_2_min) + np.square(self.mean_random_correlated_3_min))

    def calc_total_mean_uncertainty_max(self):
        """
        Calculates the maximum total mean uncertainty by adding in quadrature the uncertainties (both maximum correlated and uncorrelated).
        """
        if len(self.variogram_analysis.ranges) ==1:
            self.total_mean_uncertainty_max = np.sqrt(np.square(self.mean_random_uncorrelated) + np.square(self.mean_random_correlated_1_max))
        elif len(self.variogram_analysis.ranges) ==2:
            self.total_mean_uncertainty_max = np.sqrt(np.square(self.mean_random_uncorrelated) + np.square(self.mean_random_correlated_1_max) + np.square(self.mean_random_correlated_2_max))
        elif len(self.variogram_analysis.ranges) ==3:
            self.total_mean_uncertainty_max = np.sqrt(np.square(self.mean_random_uncorrelated) + np.square(self.mean_random_correlated_1_max) + np.square(self.mean_random_correlated_2_max) + np.square(self.mean_random_correlated_3_max))