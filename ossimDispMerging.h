//----------------------------------------------------------------------------
//
// License:  See top level LICENSE.txt file.
//
// File: openCVtestclass.h
//
// Author:  Martina Di Rita
//
// Description: Class for disparity map extraction and merging
//
//----------------------------------------------------------------------------
#ifndef ossimDispMerging_HEADER
#define ossimDispMerging_HEADER 1

//#include <opencv/cv.h>
#include <opencv2/opencv.hpp>
#include <ossim/imaging/ossimImageDataFactory.h>
#include "ossim/imaging/ossimImageHandler.h"
#include <ossim/elevation/ossimElevManager.h>
#include <ossim/base/ossimArgumentParser.h>

#include "ossimStereoPair.h"
#include "ossimOpenCvTPgenerator.h"
#include "ossimOpenCvDisparityMapGenerator.h"
#include "ossimRawImage.h"

class ossimDispMerging
{
public:
        ossimDispMerging();
        bool execute(vector<ossimStereoPair> StereoPairList, vector<ossimString> orthoListMask, vector<ossimRawImage> imageList, double currentRes, ossimArgumentParser ap, int ndisparities, int minimumDisp, int SADWindowSize, string NS, string nd, string MD, string SAD);
        cv::Mat getMergedDisparity();
        bool computeDsm(vector<ossimStereoPair> StereoPairList, ossimElevManager *elev, int b, ossimArgumentParser ap, string NS, string nd, string MD, string SAD);
        ossimRefPtr<ossimImageData> getDsm();

private:

        bool imgConversionToMat(); //apre le img ortho e diventano opencv mat
        bool imgPreProcessing(); //vedi wallis
        bool imgGetHisto(cv::Mat image, double threshold, double *minHisto, double *maxHisto);
        bool imgConversionTo8bit();
        bool disparityFusion();

        //chiamata dense matching

        vector<cv::Mat> array_metric_disp;
        //cv::Mat disparityMap;
        ossimRefPtr<ossimImageData> finalDSM;
        cv::Mat mask_ascending_tot, mask_descending_tot;
        cv::Mat merged_disp;
        cv::Mat master_mat, slave_mat;
        cv::Mat master_mat_8U,slave_mat_8U;
        ossimImageHandler* master_handler;
        ossimImageHandler* slave_handler;
        double null_disp_threshold;
        int ortho_rows, ortho_cols;
};

#endif /* #ifndef ossimDispMerging_HEADER */
