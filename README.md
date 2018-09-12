opencv_dsm
==========

OSSIM tool for automatic DSM generation and merging from a stack of optical and SAR images

==========

This is the repository for OSSIM GSoC 2016: it contains the code of DATE (Digital Automatic Terrain Extractor), the developed OSSIM plug-in for automatic Digital Surface Models (DSMs) generation and merging from a stack of optical and SAR images.

For more information about the project see:

https://summerofcode.withgoogle.com/projects/#4790138697678848

This repository only contains the new and updated files, with reference to the structure of the OSSIM repository (https://github.com/ossimlabs/ossim).

In order to compile and install this OSSIM plug-In, please observe the following instructions:

	1. Install and compile the latest OSSIM version (git clone https://github.com/ossimlabs/ossim) - Remember to set in the .bashrc file OSSIM environmental variables
	2. Install and compile the latest ossim-plugins version (turn on at least: gdal; geopdf; cnes; hdf5; opencv; potrace; sqlite; web plugins in OSSIM_DEV_HOME/ossim/cmake/scripts/ossim-cmake-config.sh) - (git clone https://github.com/ossimlabs/ossim- plugins)
	3. Open a shell in the OSSIM_DEV_HOME/ossim-plugins and copy DATE code using:
		git clone https://github.com/martidi/opencv_dsm/tree/imageStack
	4. In  OSSIM_DEV_HOME/ossim-plugins/CmakeLists.txt at line 87 add:
	   	add_subdirectory(opencv_dsm)
	5. In OSSIM_DEV_HOME/ossim/scripts run
		build.sh 
	6. In the "build" folder type
		sudo make install 
	7. If not already existing, In OSSIM_DEV_HOME create
		data → for elevation
		preferences → for preference file
	folders
	8. In order to test DATE plug-in, please create a “results” folder, containing:
		DSM
		ortho_images
		temp_elevation
		mask
	folders.
	Note that, by now, the mask folder has to contain fake metadata files (mask_name.hdr mask_name.xml mask_name_rpc.txt with the same content of raw images) in order to be able to orthorectify the masks.

	9. Put a .txt file in the “build” folder, containing the list of the images to be used their orbit (write "a" for ascending orbit and "d" for descending) and the pairs to be considered, as in the following example:
	3 %number of images to be used
	0 absolute_path_to_image_1 orbit
	1 absolute_path_to_image_2 orbit
	2 absolute_path_to_image_3 orbit

	3 %number of pairs to be processed
	0 1
	1 2
	0 2

	0 1
	1 2
	0 2

	10. From the “build” folder, run DATE:
	./bin/ossim-dsm-app input_images.txt absolute_path_to_results_folder DSM_name
	--cut-bbox-ll lat_min lon_min lat_max lon_max --meters xx --nsteps xx --projection utm --SGBM-Par ndisparities minimumDisp    		SADWindowSize
	
	
Here https://trac.osgeo.org/ossim/wiki/ACompletePhotogrammetricOSSIMtoolForAutomaticDSMenerationUsingMultiViewOpticalAndSARimages 
you can find all my 2016 GSoC reports, describing the project and the work carried out weekly.

==========

PUBBLICATIONS

[1] Di Rita, M., Nascetti, A., and Crespi M. (2018). FOSS4G DATE for DSMs generation from tri-stereo optical satellite images: development and first results, European Journal of Remote Sensing, 51:1, 472-485, DOI: 10.1080/22797254.2018.1450644

[2] Di Rita, M., Nascetti, A., and Crespi M. "Open source tool for DSMs generation from high resolution optical satellite imagery: development and testing of an OSSIM plug-in." International journal of remote sensing 38.7 (2017): 1788-1808.

[3] Di Rita, M., Nascetti, A., Fratarcangeli, F., and Crespi, M. (2016). Upgrade of FOSS DATE plug-in: implementation of a new radargrammetric DSM generation capability. The International Archives of Photogrammetry, Remote Sensing and Spatial Information Sciences, 41, 821.

[4] Di Rita, M., Nascetti, A., and Crespi, M. (2017). FOSS4G DATE ASSESSMENT ON THE ISPRS OPTICAL STEREO SATELLITE DATA: A BENCHMARK FOR DSM GENERATION. The International Archives of the Photogrammetry, Remote Sensing & Spatial Information Sciences, 42.

==========

For any doubts or issues please email me: martina.dirita@uniroma1.it


				
				
