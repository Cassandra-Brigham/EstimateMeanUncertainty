
from  uncertainty_calculation import RasterDataHandler, StatisticalAnalysis, VariogramAnalysis, UncertaintyCalculation
import numpy as np

vert_diff_path_dsm="/Users/cassandrabrigham/Documents/POSTDOC/Data analysis/Error/Code testing/Data/1st return/DoDs/test_site_5_FL_1st_return_dod.tif"
vert_diff_path_dtm="/Users/cassandrabrigham/Documents/POSTDOC/Data analysis/Error/Code testing/Data/Ground A/DoDs/test_site_5a_FL_ground_dod.tif"
output_path = "vertical_differencing_dsm_modified.tif"

#Load vertical differencing raster data from DTMs
raster_data_handler_dtm=RasterDataHandler(vert_diff_path_dtm)
raster_data_handler_dtm.load_raster()

vert_diff_array_dtm = raster_data_handler_dtm.data_array

#With the assumption that the distribution of vertical differences should be centered on zero given a large enough area, let's take the median as the measure of vertical bias.
vertical_bias = np.median(vert_diff_array_dtm)

#Get a measure of the uncertainty in the median value by bootstrap resampling (10% of the total observations, 1000x)
median_uncertainty = StatisticalAnalysis.bootstrap_uncertainty_subsample(vert_diff_array_dtm)

#Load vertical differencing raster data from DSMs
raster_data_handler_dsm=RasterDataHandler(vert_diff_path_dsm)
raster_data_handler_dsm.load_raster()

vert_diff_array_dsm = raster_data_handler_dsm.data_array

#Subtract the vertical bias from the original raster and save as a file
raster_data_handler_dsm.subtract_value_from_raster(output_path, vertical_bias)

#Create new raster handling instance for modified raster, load raster and create data array
raster2_data_handler=RasterDataHandler(output_path)
raster2_data_handler.load_raster()
vert_diff_array2 = raster2_data_handler.data_array

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

#Plot and save modified raster
fig1=raster2_data_handler.plot_raster()
fig1.savefig("outputs/modified_diff_raster.png", dpi=300)

# Plot and save stats (ground)
fig2=StatisticalAnalysis.plot_data_stats(vert_diff_array_dtm)
fig2.savefig("outputs/vert_diff_ground_stats.png", dpi=300)

# Plot and save stats (1st return)
fig3=StatisticalAnalysis.plot_data_stats(vert_diff_array_dsm)
fig3.savefig("outputs/vert_diff_1st_returns_stats.png", dpi=300)

# Plot and save variogram
fig4=V.plot_3_spherical_models_no_nugget()
fig4.savefig("outputs/variogram_with_fit.png", dpi=300)

# Write output variables to text file
file_path = "outputs/output_variables.txt"
with open(file_path, 'w') as file:
    file.write("Output variables\n")
    file.write("\tArea\n")
    file.write(f"\t\t{uncertainty.area}m^2\n")
    file.write("\tError\n")
    file.write(f"\t\tVertical bias:{vertical_bias:.4f}m\n")
    file.write(f"\t\tUncertainty in the vertical bias:{median_uncertainty:.4f}m\n")
    file.write("\tSpherical models\n")
    file.write("\t\tSpherical model 1\n")
    file.write(f"\t\t\tRange 1: {V.ranges[0]:.4f}m\n")
    file.write(f"\t\t\tSill 1: {V.sills[0]:.4f}\n")
    file.write("\t\tSpherical model 2\n")
    file.write(f"\t\t\tRange 2: {V.ranges[1]:.4f}m\n")
    file.write(f"\t\t\tSill 2: {V.sills[1]:.4f}\n")
    file.write("\t\tSpherical model 3\n")
    file.write(f"\t\t\tRange 3: {V.ranges[2]:.4f}m\n")
    file.write(f"\t\t\tSill 3: {V.sills[2]:.4f}\n")
    file.write("\tMean Uncertainty\n")
    file.write("\t\tMean, random, uncorrelated uncertainty\n")
    file.write(f"\t\t\t{uncertainty.mean_random_uncorrelated:.4f}m\n")
    file.write("\t\tMean, random, correlated uncertainty\n")
    file.write(f"\t\t\tFrom model 1:\n")
    file.write(f"\t\t\t\tOptimal:{uncertainty.mean_random_correlated_1:.4f}m\n")
    file.write(f"\t\t\t\tMinimum:{uncertainty.mean_random_correlated_1_min:.4f}m\n")
    file.write(f"\t\t\t\tMaximum:{uncertainty.mean_random_correlated_1_max:.4f}m\n")
    file.write(f"\t\t\tFrom model 2:\n")
    file.write(f"\t\t\t\tOptimal:{uncertainty.mean_random_correlated_2:.4f}m\n")
    file.write(f"\t\t\t\tMinimum:{uncertainty.mean_random_correlated_2_min:.4f}m\n")
    file.write(f"\t\t\t\tMaximum:{uncertainty.mean_random_correlated_2_max:.4f}m\n")
    file.write(f"\t\t\tFrom model 3:\n")
    file.write(f"\t\t\t\tOptimal:{uncertainty.mean_random_correlated_3:.4f}m\n")
    file.write(f"\t\t\t\tMinimum:{uncertainty.mean_random_correlated_3_min:.4f}m\n")
    file.write(f"\t\t\t\tMaximum:{uncertainty.mean_random_correlated_3_max:.4f}m\n")
    file.write("\t\tTotal mean uncertainty\n")
    file.write(f"\t\t\tOptimal:{uncertainty.total_mean_uncertainty:.4f}m\n")
    file.write(f"\t\t\tMinimum:{uncertainty.total_mean_uncertainty_min:.4f}m\n")
    file.write(f"\t\t\tMaximum:{uncertainty.total_mean_uncertainty_max:.4f}m\n")

