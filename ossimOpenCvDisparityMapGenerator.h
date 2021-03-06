//----------------------------------------------------------------------------
//
// License:  See top level LICENSE.txt file.
//
// File: ossimOpenCvDisparityMapGenerator.h
//
// Author:  Martina Di Rita
//
// Description: Class provides Disparity Map extraction
//
//----------------------------------------------------------------------------
#ifndef ossimOpenCvDisparityMapGenerator_HEADER
#define ossimOpenCvDisparityMapGenerator_HEADER 1

//#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include "ossimStereoPair.h"
#include "ossim/imaging/ossimImageHandler.h"
#include <ossim/base/ossimArgumentParser.h>

class ossimOpenCvDisparityMapGenerator
{
public:
    ossimOpenCvDisparityMapGenerator();
    //void execute(cv::Mat master_mat, cv::Mat slave_mat, ossimStereoPair StereoPair, int rows, int cols, double currentRes);
    void execute(cv::Mat master_mat, cv::Mat slave_mat, ossimStereoPair StereoPair, int rows, int cols, double currentRes, ossimArgumentParser ap, ossimImageHandler *master_handler, int ndisparities, int minimumDisp, int SADWindowSize, string NS, string nd, string MD, string SAD);
    cv::Mat getDisp();

    ossimRefPtr<ossimImageData> finalDisparity;
    cv::Mat array_disp;
    cv::Mat mergedDisp_array;
    //int ndisparities; //Maximum disparity minus minimum disparity
    //int SADWindowSize; //Matched block size
    //int minimumDisp;
};

#endif /* #ifndef ossimOpenCvDisparityMapGenerator_HEADER */
