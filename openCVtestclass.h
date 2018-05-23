//----------------------------------------------------------------------------
//
// License:  See top level LICENSE.txt file.
//
// File: openCVtestclass.h
//
// Author:  Martina Di Rita
//
// Description: Class provides OpenCV functions for DSM extraction
//
//----------------------------------------------------------------------------
#ifndef openCVtestclass_HEADER
#define openCVtestclass_HEADER 1

//#include <opencv/cv.h>
#include <opencv2/opencv.hpp>

//https://stackoverflow.com/questions/33400823/opencv-3-0-0-cvvector-missing
namespace cv
{
    using std::vector;
}

class openCVtestclass
{
public:
	openCVtestclass();
	openCVtestclass(ossimRefPtr<ossimImageData> master, ossimRefPtr<ossimImageData> slave); 
	bool execute();
	bool writeDisparity(double mean_conversionF);
    ossimRefPtr<ossimImageData> computeDSM(double mean_conversionF, ossimElevManager* elev, ossimImageGeometry* master_geom);
	cv::Mat wallis(cv::Mat raw_image);

	cv::Mat master_mat, slave_mat;
	cv::vector<cv::KeyPoint> keypoints1, keypoints2;
	vector<cv::DMatch > good_matches;
	cv::Mat out_disp; 
	double null_disp_threshold;

};

#endif /* #ifndef openCVtestclass_HEADER */