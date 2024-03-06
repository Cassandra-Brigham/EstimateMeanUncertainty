# EstimateMeanUncertainty

## Pseudo-code guide for uncertainty calculation



This document outlines a mean uncertainty calculation that follows the vertical differencing workflow. We envision that this tool can be added to all the vertical differencing, including with OT datasets, USGS-USGS datasets, and NOAA-NOAA datasets, for the cases where we start the differencing with the point cloud and with the raster. The one condition is the datasets need to be classified (at a minimum, ground points are labeled). 



We divide the steps into two cases: 1) the user selects the “Ground only” option when initiating the vertical differencing workflow (DTM case); 2) the user does not select that option (DSM case). 



If the user is creating a vertical differencing raster from DTMs, use the script “run_uncertainty_calculation_DTMs.py” (Case 1). If using DSMs, use “run_uncertainty_calculation_DSMs.py” (Case 2).  



### Case 1: Vertical differencing raster from DTMs (ground points only)



#### Step 1.

The required Python libraries are listed in the environment.yml file. 

It uses a Cython module (“variogram_tools.pyx”), so there are a few steps to compile and build the module. 

Install Cython:



% pip install cython



Run “setup.py”:



% python setup.py build_ext –inplace



For case 1, we will use the script “run_uncertainty_calculation_DTMs.py.” Currently, I have the output figures and text file going to an “outputs/” folder in the same directory. 

To change that, modify their paths in the script here:

![Image1](/README_files/image1.png)



 

#### Step 2. 

For case 1, use the script “run_uncertainty_calculation_DTMs.py.” Its inputs are: 1) the path to the vertical differencing raster (“vertical_differencing_dtm.tif”) generated during the OT Vertical Differencing algorithm; 2) the path to an output raster that will be generated when the script is run (“vertical_differencing_dtm_modified.tif”); 3) the unit of the raster (“m”); 4) the resolution of the vertical differencing raster (1.0).



The script can be run directly from the command line.  



You can either write the arguments directly into the command line:



% python run_uncertainty_calculation_DTMs.py "vertical_differencing_dtm.tif" "vertical_differencing_dtm_modified.tif" "m" 1.0   



Or you can write out the arguments (each argument on a separate line) in a text file (“arguments_DTMs.txt”) in the same folder:

![Image2](/README_files/image2.png)





And redirect those inputs with xargs:



% cat arguments.txt | xargs python run_uncertainty_calculation_DTMs.py



Many of the modules used in this script are stored in the “uncertainty_calculation.py” script, which defines classes to handle raster data, conduct statistical analyses, create, fit, and plot variograms and calculate the uncertainty parameters. 



Make sure that “uncertainty_calculation.py” is in the same directory as “run_uncertainty_calculation_DTMs.py.”





#### Step 3.

There are three output figures (“modified_diff_raster.png,” “vert_diff_ground_stats.png,” “variogram_with_fit.png”) and one text file (“output_variables.txt”). The text file contains the uncertainty values, as well as some other important contextual information. Below is an example of the outputs:



![Image3](/README_files/image3.png)

“modified_diff_raster.png” : Modified vertical differencing raster (vertical bias has been subtracted). 

![Image4](/README_files/image4.png)

“vert_diff_ground_stats.png” :  Histogram of vertical differencing results with exploratory statistics

![Image5](/README_files/image5.png)

“variogram_with_fit.png” : Empirical mean variogram with spread (blue) and fitted model (red) with RMSE value. Optimal range values are plotted as red, green and blue dashed vertical lines. Pairwise measurement count per lag is shown in histogram in yellow above. 


![Image6](/README_files/image6.png)

“output_variables.txt”: Text file with area of interest, vertical bias, variogram modelling results, and uncertainty calculation results.



### Case 2: Vertical differencing raster from DSMs



#### Step 0.

This is the case that the user does not select “Ground only” while setting up their differencing job in OT. By default, DSMs of the reference and compare point cloud datasets will be made and used to generate a vertical differencing raster (“vertical_differencing_dsm.tif”) during the OT Vertical Differencing algorithm. 

In parallel to those DSMs, make DTMs (using only ground points) of the reference and compare datasets (same extents as DSMs) and generate a vertical differencing raster using those DTMs (“vertical_differencing_dtm.tif”).



#### Step 1.

The required Python libraries are listed in the environment.yml file. 

Compile and build the Cython module. 

Install Cython:



% pip install cython



Run “setup.py”:



% python setup.py build_ext --inplace



For case 1, we will use the script “run_uncertainty_calculation_DSMs.py.” Currently, I have the output figures and text file going to an “outputs/” folder in the same directory. 

To change that, modify their paths in the script here: 

![Image7](/README_files/image7.png)





#### Step 2. 

For case 2, use the script “run_uncertainty_calculation_DSMs.py.” Its inputs are: 1) the path to the DSM-derived vertical differencing raster(“vertical_differencing_dsm.tif”); 2) the path to the DTM-derived vertical differencing raster (“vertical_differencing_dtm.tif”); 3) the path to an output raster that will be generated when the script is run (“vertical_differencing_dsm_modified.tif”); 4) the unit of the raster (“m”); 5) the resolution of the vertical differencing raster (1.0).



Run the script from the command line.  



You can either write the arguments directly into the command line:



% python run_uncertainty_calculation_DSMs.py "vertical_differencing_dsm.tif" "vertical_differencing_dtm.tif" "vertical_differencing_dsm_modified.tif" "m" 1.0   



Or you can write out the arguments (each argument on a separate line) in a text file (“arguments_DSMs.txt”) in the same folder:

![Image8](/README_files/image8.png)

And redirect those inputs with xargs:



% cat arguments_DSMs.txt | xargs python run_uncertainty_calculation_DSMs.py



Again, make sure that “uncertainty_calculation.py” is in the same directory as “run_uncertainty_calculation_DSMs.py.”



#### Step 3.

There are four output figures (“modified_diff_raster.png,” “vert_diff_1st_returns_stats.png,” “vert_diff_ground_stats.png,” “variogram_with_fit.png”) and one text file (“output_variables.txt”). The text file contains the uncertainty values, as well as some other important contextual information. Below is the same example location, but outputs from the DSM differencing results:



![Image9](/README_files/image9.png)

“modified_diff_raster.png” : Modified vertical differencing raster (vertical bias has been subtracted from DSM-derived differencing raster). 





![Image10](/README_files/image10.png)

“vert_diff_1st_returns_stats.png” : Histogram of DSM-derived vertical differencing results with exploratory statistics



![Image11](/README_files/image11.png)

“vert_diff_ground_stats.png” : Histogram of DTM-derived vertical differencing results with exploratory statistics


![Image12](/README_files/image12.png)

“variogram_with_fit.png” : Empirical mean variogram with spread (blue) and fitted model (red) with RMSE value. Optimal range values are plotted as red, green and blue dashed vertical lines. Pairwise measurement count per lag is shown in histogram in yellow above. 


![Image13](/README_files/image13.png)

“output_variables.txt” : Screenshot of text file with area of interest, vertical bias, variogram modelling results, and uncertainty calculation results



### Notes

Eventually, the units and the resolution should update dynamically based on the dataset metadata and the DEM resolution selected by the user.
