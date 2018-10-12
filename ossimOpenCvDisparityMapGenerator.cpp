//----------------------------------------------------------------------------
//
// License:  See top level LICENSE.txt file.
//
// File: ossimOpenCvDisparityMapGenerator.cpp
//
// Author:  Martina Di Rita
//
// Description: Class providing Disparity Map generation
//
//----------------------------------------------------------------------------

#include <ossim/imaging/ossimImageSource.h>
#include <ossim/imaging/ossimMemoryImageSource.h>
#include "ossim/imaging/ossimImageFileWriter.h"
#include "ossim/imaging/ossimImageWriterFactoryRegistry.h"
#include "ossimOpenCvDisparityMapGenerator.h"

#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
// Note: These are purposely commented out to indicate non-use.
// #include <opencv2/nonfree/nonfree.hpp>
// #include <opencv2/nonfree/features2d.hpp>
// Note: These are purposely commented out to indicate non-use.
#include <iostream>
#include <ossim/base/ossimTimer.h>

ossimOpenCvDisparityMapGenerator::ossimOpenCvDisparityMapGenerator()
{
	
}

//void ossimOpenCvDisparityMapGenerator::execute(cv::Mat master_mat, cv::Mat slave_mat, ossimStereoPair StereoPair, int rows, int cols, double currentRes)
void ossimOpenCvDisparityMapGenerator::execute(cv::Mat master_mat, cv::Mat slave_mat, ossimStereoPair StereoPair, int rows, int cols, double currentRes, ossimArgumentParser ap, ossimImageHandler* master_handler, int ndisparities, int minimumDisp, int SADWindowSize, string NS, string nd, string MD, string SAD)
{
	cout << "DISPARITY MAP GENERATION \t in progress..." << endl;
	ossimTimer::instance()->setStartTick();
	//******************************************************
	// Abilitate for computing disparity on different scales 
	//******************************************************
	/*
	double fscale = 1.0/2.0;
	cv::resize(master_mat, master_mat, cv::Size(), fscale, fscale, cv::INTER_AREA );
	cv::resize(slave_mat, slave_mat, cv::Size(), fscale, fscale, cv::INTER_AREA );	
	cv::namedWindow( "Scaled master", CV_WINDOW_NORMAL );
	cv::imshow( "Scaled master", master_mat);
	cv::namedWindow( "Scaled slave", CV_WINDOW_NORMAL );
	cv::imshow( "Scaled slave", slave_mat);
	*/	


    // esporre parametri o trovare un modo per rendere i parametri automatici
    //ndisparities = 64; //Maximum disparity minus minimum disparity - prove a 128  64
    //minimumDisp = -16; //prova a    -16
    //SADWindowSize = 5; //Matched block size

    // Disparity Map generation
    int cn = master_mat.channels();
    

    /*cv::StereoSGBM sgbm;
    sgbm.preFilterCap = 63;
    sgbm.SADWindowSize = SADWindowSize > 0 ? SADWindowSize : 3;
    sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
    sgbm.P2 = 40*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
    sgbm.minDisparity = minimumDisp; // Minimum possible disparity value  //con fattore di conversione 1 metti -16*2
    sgbm.numberOfDisparities = ndisparities;
    sgbm.uniquenessRatio = 5;
    sgbm.speckleWindowSize = 100;
    sgbm.speckleRange = 1;
    sgbm.disp12MaxDiff = 1; // Maximum allowed difference (in integer pixel units) in the left-right disparity check
    //sgbm.fullDP = true; //activate for consider 8 directions (Hirschmuller algorithm) instead of 5;*/
    cout << ndisparities << " " << minimumDisp <<" "<< SADWindowSize << " function parameters " << endl<<endl;

    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(minimumDisp, ndisparities, SADWindowSize);
    sgbm->setPreFilterCap(63);
    sgbm->setBlockSize (SADWindowSize > 0 ? SADWindowSize : 3);
    sgbm->setP1 (8*cn*sgbm->getBlockSize()*sgbm->getBlockSize());
    sgbm->setP2 (40*cn*sgbm->getBlockSize()*sgbm->getBlockSize());//64
    sgbm->setMinDisparity (minimumDisp); // Minimum possible disparity value  //con fattore di conversione 1 metti -16*2
    sgbm->setNumDisparities ( ndisparities);
    sgbm->setUniquenessRatio( 5);
    sgbm->setSpeckleWindowSize (100);
    sgbm->setSpeckleRange ( 1);
    sgbm->setDisp12MaxDiff ( 1); // Maximum allowed difference (in integer pixel units) in the left-right disparity check
    //sgbm->fullDP = true; //activate for consider 8 directions (Hirschmuller algorithm) instead of 5;


    double minVal, maxVal;

    cv::Mat array_disp_8U;
    //sgbm(master_mat, slave_mat, array_disp);
    //OPENCV3 //https://docs.opencv.org/3.4/d1/d9f/classcv_1_1stereo_1_1StereoBinarySGBM.html
    sgbm->compute (master_mat, slave_mat, array_disp);
    cv::imwrite( "float_Disparity_0.tif", array_disp);

    cv::FileStorage fs("test.yml", cv::FileStorage::WRITE);
    fs << "cameraMatrix" << array_disp;
    fs.release();

    minMaxLoc( array_disp, &minVal, &maxVal );
    array_disp.convertTo( array_disp_8U, CV_8UC1, 255/(maxVal - minVal), -minVal*255/(maxVal - minVal));
    
    cout << "elapsed time in seconds: " << std::setiosflags(ios::fixed) << std::setprecision(3) << ossimTimer::instance()->time_s() << endl << endl;

    cout << "min\t" << minVal << " " << "max\t" << maxVal << endl;
    cv::namedWindow( "SGM Disparity", cv::WINDOW_NORMAL );
    cv::imshow( "SGM Disparity", array_disp_8U);
    cv::imwrite("SGM Disparity.png", array_disp_8U);
    
    //ossimFilename pathDISP =  ossimFilename(ap[2]) + ossimString("disparity/") ossimFilename(ap[3]) + ossimString("SGM_Disparity_ns_") + NS + ossimString("_nd_") + nd + ossimString("_MD_") + MD + ossimString("_SAD_") + SAD + ossimString(".tif");

    ostringstream convert5, convert6; //// stream used for the conversion
    convert5 << ap[2];
    convert6 << ap[3];
    string path1 = convert5.str();
    string path2 = convert6.str();
    cv::imwrite(path1 + "disparity/" + path2 + "_SGM_Disparity_ns_" + NS + "_nd_" + nd + "_MD_" + MD + "_SAD_" + SAD + ".tif", array_disp_8U);
 
    //cv::waitKey(0);


	//******************************************************
	// Abilitate for computing disparity on different scales 
	//****************************************************** 	
	//array_disp = array_disp/fscale; // to consider the scale factor also in the disparity values (i singoli valori sono alterati)
	//cv::resize(array_disp, array_disp, cv::Size(), 1.0/fscale, 1.0/fscale, cv::INTER_AREA ); // to resize the disparity map as the initial image

	//Create and write the log file
	ofstream disparity;
	disparity.open ("DSM_parameters_disparity.txt");
	disparity <<"DISPARITY RANGE:" << " " << ndisparities << endl;
	disparity <<"SAD WINDOW SIZE:" << " " << SADWindowSize<< endl;
	disparity << "MINIMUM DISPARITY VALUE:"<< sgbm->getMinDisparity() << endl;
    disparity.close();

    // ogni disp map deve essere ruotata, convertita a CV_64F, divisa per 16 bit, resa metrica tramite il conv fact

    // Rotation for along-track OPTICAL images
    //********* To be commented for SAR images *********
    //cv::transpose(array_disp, array_disp);
    //cv::flip(array_disp, array_disp, 0);
    //********* To be commented for SAR images *********


    //double angle = -13; // - senso orario; + senso antiorario
    // get rotation matrix for rotating the image around its center
    cv::Point2f center(array_disp.cols/2.0, array_disp.rows/2.0);
    cv::Mat rot = cv::getRotationMatrix2D(center,  StereoPair.getMeanRotationAngle(), 1.0);

    // determine bounding rectangle
    cv::Rect bbox = cv::RotatedRect(center,array_disp.size(),  StereoPair.getMeanRotationAngle()).boundingRect();
    // adjust transformation matrix
    rot.at<double>(0,2) += bbox.width/2.0 - center.x;
    rot.at<double>(1,2) += bbox.height/2.0 - center.y;

    cv::warpAffine(array_disp, array_disp, rot, bbox.size());

    //resize for rotation
    // prendo le dimensioni della ortho, mi metto al centro dell'immagine etaglio a dx, sx, sopra e sotto della quantità richiesta
    int xTopLeft = (bbox.width - cols)/2;
    int yTopLeft = (bbox.height - rows)/2;
    // Setup a rectangle to define your region of interest
    cv::Rect myROI(xTopLeft, yTopLeft, cols, rows);

    // Crop the full image to that image contained by the rectangle myROI
    // Note that this doesn't copy the data
    array_disp = array_disp(myROI);
    cv::imwrite("rotated_disp_a.tiff", array_disp);





   array_disp.convertTo(array_disp, CV_64F);
   array_disp = currentRes*((array_disp/16.0)) / StereoPair.getConversionFactor(); //quando divido per il fattore di conversione le rendo metriche
   /*cv::Scalar tempVal = mean( array_disp );
   float myMAtMean = tempVal.val[0];
   cout << "media "<< myMAtMean << endl;
   cout << "conv " << StereoPair.getConversionFactor() << endl;*/
   cout << "risoluzione " << currentRes << endl;
   cout << "fattore di conv " << StereoPair.getConversionFactor() << endl;

    for(int i=0; i< array_disp.rows; i++)
    {
        for(int j=0; j< array_disp.cols; j++)
        {
            if(array_disp.at<double>(i,j) < (minimumDisp + 5.0 - 1 )/ StereoPair.getConversionFactor())
            {
                array_disp.at<double>(i,j) = -9999.0;
            }
        }
    }

    cv::FileStorage valori("valori.yml", cv::FileStorage::WRITE);
    valori << "cameraMatrix" << array_disp;
    valori.release();


    // DISPARITY MAP GEOREFERENCE
    // Set the destination image size:
   ossimIpt image_size (array_disp.cols , array_disp.rows);
    finalDisparity = ossimImageDataFactory::instance()->create(0, OSSIM_FLOAT32, 1, image_size.x, image_size.y);

    if(finalDisparity.valid())
       finalDisparity->initialize();
    // else
    //  return -1;

    for (int i=0; i< array_disp.cols; i++) // for every column
    {
        for(int j=0; j< array_disp.rows; j++) // for every row
        {
            finalDisparity->setValue(i,j,array_disp.at<double>(j,i));
        }
    }

    // Create output image chain:
    ossimFilename pathDISP = ossimFilename(ap[2]) + ossimString("disparity/") + ossimFilename(ap[3]) + ossimString("_Disparity_ns_") + NS + ossimString("_nd_") + nd + ossimString("_MD_") + MD + ossimString("_SAD_") + SAD + ossimString(".TIF");
    ossimImageGeometry* master_geom = master_handler->getImageGeometry().get();
    ossimRefPtr<ossimMemoryImageSource> memSource = new ossimMemoryImageSource;
    memSource->setImage(finalDisparity);
    memSource->setImageGeometry(master_geom);
    cout << "disparity map size " << master_geom->getImageSize() << endl;
    memSource->saveImageGeometry();

    ossimImageFileWriter* writer = ossimImageWriterFactoryRegistry::instance()->createWriter(pathDISP);
    writer->connectMyInputTo(0, memSource.get());
    writer->execute();
    writer->close();
    writer = 0;
    memSource = 0;

}



cv::Mat ossimOpenCvDisparityMapGenerator::getDisp()
{

    return array_disp;

}
