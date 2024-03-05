# variogram_tools.pyx
# cython: language_level=3


#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt
from libc.stdlib cimport qsort
from libc.string cimport memset

# Function to calculate bin width and the range of distances
cpdef calculate_bin_width(cnp.ndarray[cnp.float64_t, ndim=2] coords, int n_bins_input):
    cdef int n_samples = coords.shape[0]
    cdef double max_distance = 0, min_distance = float('inf'), distance
    cdef int i, j

    # Find the minimum and maximum distances between points
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            distance = sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2)
            if distance > max_distance:
                max_distance = distance
            if distance < min_distance:
                min_distance = distance

    # Calculate and return the bin width and the min/max distances
    cdef double bin_width = (max_distance - min_distance) / (n_bins_input - 1)
    return bin_width, min_distance, max_distance

cpdef dowd_estimator_cy(cnp.ndarray[cnp.float64_t, ndim=2] coords, cnp.ndarray[cnp.float64_t, ndim=1] values, int n_bins_input, double bin_width, double min_distance, double max_distance):
    cdef int n_bins = n_bins_input - 1
    cdef cnp.ndarray[cnp.float64_t, ndim=1] variogram = np.zeros(n_bins, dtype=np.float64)
    cdef cnp.ndarray[cnp.intp_t, ndim=1] count = np.zeros(n_bins, dtype=np.intp)  # For counting samples in each bin
    cdef int i, j, k
    cdef double distance, difference
    cdef list squared_differences = [list() for _ in range(n_bins)]  # List of lists to store squared differences for each bin

    # Calculate squared differences
    for i in range(coords.shape[0]):
        for j in range(i + 1, coords.shape[0]):
            distance = sqrt((coords[i, 0] - coords[j, 0])**2 + (coords[i, 1] - coords[j, 1])**2)
            difference = (values[i] - values[j])**2

            if distance >= min_distance and distance < max_distance:
                k = int((distance - min_distance) / bin_width)
                if k >= n_bins:  # Handle rounding issues
                    k = n_bins - 1
                squared_differences[k].append(difference)
                count[k] += 1  # Increment count for this bin

    # Calculate variogram using Dowd's estimator
    for i in range(n_bins):
        if squared_differences[i]:
            median_difference = np.median(squared_differences[i])
            variogram[i] = 2.198 * (median_difference)/2

 

    return variogram,count