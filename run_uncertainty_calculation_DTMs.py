import numpy as np
from  uncertainty_calculation import RasterDataHandler, StatisticalAnalysis, VariogramAnalysis, UncertaintyCalculation
import os
import argparse


def run_uncertainty_calculation_DTMs (vert_diff_path_dtm, output_path, unit, dem_resolution):
    #Load raster data
    raster_data_handler_dtm=RasterDataHandler(vert_diff_path_dtm, unit, dem_resolution)
    raster_data_handler_dtm.load_raster()

    vert_diff_array_dtm = raster_data_handler_dtm.data_array

    #With the assumption that the distribution of vertical differences should be centered on zero given a large enough area, let's take the median as the measure of vertical bias.
    vertical_bias = np.median(vert_diff_array_dtm)

    #Get a measure of the uncertainty in the median value by bootstrap resampling (10% of the total observations, 1000x)
    stats_dtm = StatisticalAnalysis(raster_data_handler_dtm)
    median_uncertainty = stats_dtm.bootstrap_uncertainty_subsample()

    #Subtract the vertical bias from the original raster and save as a file
    raster_data_handler_dtm.subtract_value_from_raster(output_path, vertical_bias)

    #Create new raster handling instance for modified raster, load raster and create data array
    raster2_data_handler=RasterDataHandler(output_path, unit, dem_resolution)
    raster2_data_handler.load_raster()

    #Create variogram analysis instance based on modified raster
    V = VariogramAnalysis(raster2_data_handler)

    #Calculate a mean variogram with 75 bins from variograms made over 10 runs
    V.calculate_mean_variogram_numba(area_side = 250, samples_per_area = 500, max_samples = 100000, bin_width = 30, max_n_bins = 3000, n_runs = 30, cell_size = 50, n_offsets = 100, max_lag_multiplier = 1/2, normal_transform = False, weights = False)

    #Fit a sum of up to three spherical models to the mean empirical variogram
    V.fit_best_spherical_model()
    
    #Get the number of models and nugget value, if exists
    n_models = len(V.ranges)
    nugget = V.best_nugget

    #Create an instance to calculate uncertainty values from raster data and variography
    uncertainty=UncertaintyCalculation(V)

    #Calculate mean, random, uncorrelated uncertainty
    uncertainty.calc_mean_random_uncorrelated()

    #Calculate mean, random, correlated uncertainty, for optimal, minimum and maximum ranges and sills 
    uncertainty.calc_mean_random_correlated()
    uncertainty.calc_mean_random_correlated_min()
    uncertainty.calc_mean_random_correlated_max()

    #Calculate total mean uncertainty
    uncertainty.calc_total_mean_uncertainty()
    uncertainty.calc_total_mean_uncertainty_min()
    uncertainty.calc_total_mean_uncertainty_max()

    #Create output folder
    if not os.path.exists("outputs/"):
        os.makedirs("outputs/")

    #Plot and save modified raster
    fig1=raster2_data_handler.plot_raster("Vertical differencing results corrected for vertical bias",normal_transform=False)
    fig1.savefig("outputs/modified_diff_raster_ground.png", dpi=300)

    # Plot and save stats
    fig2=stats_dtm.plot_data_stats(normal_transform = False, filtered = True)
    fig2.savefig("outputs/vert_diff_ground_stats_ground.png", dpi=300)

    # Plot and save variogram
    fig3=V.plot_best_spherical_model()
    fig3.savefig("outputs/variogram_with_fit_ground.png", dpi=300)

    # Write output variables to text file
    file_path = "outputs/uncertainty_estimation_output_ground.txt"  
    with open(file_path, 'w') as file:
        file.write("Output variables\n")
        file.write("\tArea\n")
        file.write(f"\t\t{len(vert_diff_array_dtm)}" + unit + "^2\n")
        file.write("\tError\n")
        file.write(f"\t\tVertical bias: {vertical_bias:.3f}" + unit + "\n")
        file.write(f"\t\tUncertainty in the vertical bias: {median_uncertainty:.3f}" + unit + "\n")
        file.write("\tSpherical models\n")
        
        for i in range(len(V.ranges)):  # Dynamically iterate through the models
            file.write(f"\t\tSpherical model {i + 1}\n")
            file.write(f"\t\t\tRange {i + 1}: {V.ranges[i]:.3f}" + unit + "\n")
            file.write(f"\t\t\tSill {i + 1}: {V.sills[i]:.3f}\n")
        
        file.write("\tMean Uncertainty\n")
        file.write("\t\tMean, random, uncorrelated uncertainty\n")
        file.write(f"\t\t\t{uncertainty.mean_random_uncorrelated:.3f}" + unit + "\n")
        
        file.write("\t\tMean, random, correlated uncertainty\n")
        for i in range(len(V.ranges)):  # Use the same number of models as before
            file.write(f"\t\t\tFrom model {i + 1}:\n")
            file.write(f"\t\t\t\tOptimal: {getattr(uncertainty, f'mean_random_correlated_{i + 1}'): .3f}" + unit + "\n")
            file.write(f"\t\t\t\tMinimum: {getattr(uncertainty, f'mean_random_correlated_{i + 1}_min'): .3f}" + unit + "\n")
            file.write(f"\t\t\t\tMaximum: {getattr(uncertainty, f'mean_random_correlated_{i + 1}_max'): .3f}" + unit + "\n")
        
        file.write("\t\tTotal mean uncertainty\n")
        file.write(f"\t\t\tOptimal: {uncertainty.total_mean_uncertainty:.3f}" + unit + "\n")
        file.write(f"\t\t\tMinimum: {uncertainty.total_mean_uncertainty_min:.3f}" + unit + "\n")
        file.write(f"\t\t\tMaximum: {uncertainty.total_mean_uncertainty_max:.3f}" + unit + "\n")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A calculation to estimate total mean uncertainty and its constituent components for vertical differencing rasters derived from Digital Terrain Models.')
    parser.add_argument('vert_diff_path_dtm', type=str, help='DTM path')
    parser.add_argument('output_path', type=str, help='Path to raster of vertical differencing results minus the vertical bias')
    parser.add_argument('unit', type=str, help='Units of input rasters')
    parser.add_argument('dem_resolution', type=float, help='Resolution of DTM rasters')
    args = parser.parse_args()
    
    run_uncertainty_calculation_DTMs(args.vert_diff_path_dtm, args.output_path, args.unit, args.dem_resolution)