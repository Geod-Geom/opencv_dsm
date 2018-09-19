//----------------------------------------------------------------------------
//
// License:  See top level LICENSE.txt file.
//
// File: ossimOpenCvTPgenerator.h
//
// Author:  Martina Di Rita
//
// Description: Class provides a TPs generator
//
//----------------------------------------------------------------------------
#ifndef ossimOpenCvTPgenerator_HEADER
#define ossimOpenCvTPgenerator_HEADER 1

//#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <ossim/base/ossimRefPtr.h>
#include <ossim/imaging/ossimImageDataFactory.h>

//https://stackoverflow.com/questions/33400823/opencv-3-0-0-cvvector-missing
namespace cv
{
    using std::vector;
}

class ossimOpenCvTPgenerator
{
public:
    ossimOpenCvTPgenerator();
    ossimOpenCvTPgenerator(cv::Mat master, cv::Mat slave);

    bool operator()(const cv::KeyPoint& a, const cv::KeyPoint& b);
    // running command
    bool execute();
    cv::Mat getWarpedImage();

    // private function
    void TPgen();
    void TPdraw();
    cv::Mat estRT(std::vector<cv::Point2f> master, std::vector<cv::Point2f> slave);
    bool TPwarp();

    cv::Mat mask_master, mask_slave;
    cv::Mat slave_mat_warp;
    cv::Mat master_mat, slave_mat;
    cv::vector<cv::KeyPoint> keypoints1, keypoints2;
    vector<cv::DMatch > good_matches;
    double slave_x, slave_y, master_x, master_y;
};

#endif /* #ifndef ossimOpenCvTPgenerator_HEADER */