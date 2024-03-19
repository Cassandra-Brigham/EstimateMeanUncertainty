
from  uncertainty_calculation import RasterDataHandler, StatisticalAnalysis, VariogramAnalysis, UncertaintyCalculation
import numpy as np
import os
import argparse

def run_uncertainty_calculation_DSMs (vert_diff_path_dsm, vert_diff_path_dtm, output_path, unit, dem_resolution):
    #Load vertical differencing raster data from DTMs
    raster_data_handler_dtm=RasterDataHandler(vert_diff_path_dtm, unit, dem_resolution)
    raster_data_handler_dtm.load_raster()

    vert_diff_array_dtm = raster_data_handler_dtm.data_array

    #With the assumption that the distribution of vertical differences should be centered on zero given a large enough area, let's take the median as the measure of vertical bias.
    vertical_bias = np.median(vert_diff_array_dtm)

    #Get a measure of the uncertainty in the median value by bootstrap resampling (10% of the total observations, 1000x)
    stats_dtm = StatisticalAnalysis(raster_data_handler_dtm)
    median_uncertainty = stats_dtm.bootstrap_uncertainty_subsample()

    #Load vertical differencing raster data from DSMs
    raster_data_handler_dsm=RasterDataHandler(vert_diff_path_dsm,unit,dem_resolution)
    raster_data_handler_dsm.load_raster()

    #Subtract the vertical bias from the original raster and save as a file
    raster_data_handler_dsm.subtract_value_from_raster(output_path, vertical_bias)

    #Create new raster handling instance for modified raster, load raster and create data array
    raster2_data_handler=RasterDataHandler(output_path, unit, dem_resolution)
    raster2_data_handler.load_raster()

    #Create variogram analysis instance based on modified raster
    V = VariogramAnalysis(raster2_data_handler)

    #Calculate a mean variogram with 75 bins from variograms made over 10 runs
    V.calculate_mean_variogram(75,10)

    #Fit a sum of three spherical models to the mean empirical variogram
    V.fit_3_spherical_models_no_nugget()

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
    fig1=raster2_data_handler.plot_raster()
    fig1.savefig("outputs/modified_diff_raster_1st_return.png", dpi=300)

    # Plot and save stats (ground)
    fig2=stats_dtm.plot_data_stats()
    fig2.savefig("outputs/vert_diff_ground_stats_1st_return.png", dpi=300)

    # Plot and save stats (1st return)
    stats_dsm = StatisticalAnalysis(raster_data_handler_dsm)
    fig3=stats_dsm.plot_data_stats()
    fig3.savefig("outputs/vert_diff_1st_returns_stats_1st_return.png", dpi=300)

    # Plot and save variogram
    fig4=V.plot_3_spherical_models_no_nugget()
    fig4.savefig("outputs/variogram_with_fit_1st_return.png", dpi=300)

    
    # Write output variables to text file
    file_path = "outputs/output_variables.txt"
    with open(file_path, 'w') as file:
        file.write("Output variables\n")
        file.write("\tArea\n")
        file.write(f"\t\t{uncertainty.area}"+unit+"^2\n")
        file.write("\tError\n")
        file.write(f"\t\tVertical bias:{vertical_bias:.4f}"+unit+"\n")
        file.write(f"\t\tUncertainty in the vertical bias:{median_uncertainty:.4f}"+unit+"\n")
        file.write("\tSpherical models\n")
        file.write("\t\tSpherical model 1\n")
        file.write(f"\t\t\tRange 1: {V.ranges[0]:.4f}"+unit+"\n")
        file.write(f"\t\t\tSill 1: {V.sills[0]:.4f}\n")
        file.write("\t\tSpherical model 2\n")
        file.write(f"\t\t\tRange 2: {V.ranges[1]:.4f}"+unit+"\n")
        file.write(f"\t\t\tSill 2: {V.sills[1]:.4f}\n")
        file.write("\t\tSpherical model 3\n")
        file.write(f"\t\t\tRange 3: {V.ranges[2]:.4f}"+unit+"\n")
        file.write(f"\t\t\tSill 3: {V.sills[2]:.4f}\n")
        file.write("\tMean Uncertainty\n")
        file.write("\t\tMean, random, uncorrelated uncertainty\n")
        file.write(f"\t\t\t{uncertainty.mean_random_uncorrelated:.4f}"+unit+"\n")
        file.write("\t\tMean, random, correlated uncertainty\n")
        file.write(f"\t\t\tFrom model 1:\n")
        file.write(f"\t\t\t\tOptimal:{uncertainty.mean_random_correlated_1:.4f}"+unit+"\n")
        file.write(f"\t\t\t\tMinimum:{uncertainty.mean_random_correlated_1_min:.4f}"+unit+"\n")
        file.write(f"\t\t\t\tMaximum:{uncertainty.mean_random_correlated_1_max:.4f}"+unit+"\n")
        file.write(f"\t\t\tFrom model 2:\n")
        file.write(f"\t\t\t\tOptimal:{uncertainty.mean_random_correlated_2:.4f}"+unit+"\n")
        file.write(f"\t\t\t\tMinimum:{uncertainty.mean_random_correlated_2_min:.4f}"+unit+"\n")
        file.write(f"\t\t\t\tMaximum:{uncertainty.mean_random_correlated_2_max:.4f}"+unit+"\n")
        file.write(f"\t\t\tFrom model 3:\n")
        file.write(f"\t\t\t\tOptimal:{uncertainty.mean_random_correlated_3:.4f}"+unit+"\n")
        file.write(f"\t\t\t\tMinimum:{uncertainty.mean_random_correlated_3_min:.4f}"+unit+"\n")
        file.write(f"\t\t\t\tMaximum:{uncertainty.mean_random_correlated_3_max:.4f}"+unit+"\n")
        file.write("\t\tTotal mean uncertainty\n")
        file.write(f"\t\t\tOptimal:{uncertainty.total_mean_uncertainty:.4f}"+unit+"\n")
        file.write(f"\t\t\tMinimum:{uncertainty.total_mean_uncertainty_min:.4f}"+unit+"\n")
        file.write(f"\t\t\tMaximum:{uncertainty.total_mean_uncertainty_max:.4f}"+unit+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='A calculation to estimate total mean uncertainty and its constituent components for vertical differencing rasters derived from Digital Surface Models.')
    parser.add_argument('vert_diff_path_dsm', type=str, help='DSM path')
    parser.add_argument('vert_diff_path_dtm', type=str, help='DTM path')
    parser.add_argument('output_path', type=str, help='Path to raster of vertical differencing results minus the vertical bias')
    parser.add_argument('unit', type=str, help='Units of input rasters')
    parser.add_argument('dem_resolution', type=float, help='Resolution of DSM rasters')
    args = parser.parse_args()
    
    run_uncertainty_calculation_DSMs(args.vert_diff_path_dsm, args.vert_diff_path_dtm, args.output_path, args.unit, args.dem_resolution)


