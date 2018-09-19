//----------------------------------------------------------------------------
//
// License:  See top level LICENSE.txt file.
//
// File: ossimOpenCvTPgenerator.cpp
//
// Author:  Martina Di Rita
//
// Description: Class providing a TPs generator
//
//----------------------------------------------------------------------------

#include <ossim/imaging/ossimImageSource.h>
#include "ossimOpenCvTPgenerator.h"
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
// Note: These are purposely commented out to indicate non-use.
// #include <opencv2/nonfree/nonfree.hpp>
// #include <opencv2/nonfree/features2d.hpp>
// Note: These are purposely commented out to indicate non-use.
#include <vector>
#include <iostream>

ossimOpenCvTPgenerator::ossimOpenCvTPgenerator()
{

}


ossimOpenCvTPgenerator::ossimOpenCvTPgenerator(cv::Mat master, cv::Mat slave)
{
    // Create the OpenCV images
    master_mat = master;
    slave_mat = slave;
}

bool ossimOpenCvTPgenerator::execute()
{

    TPgen();
    TPdraw();
    TPwarp();

    //cv::Mat slave_mat_warp = TPfinder->warp(slave_mat);
   // slave_mat_warp = TPfinder->warp(slave_mat);

    //spostao in dispMapGen
    /*
    ossimOpenCvDisparityMapGenerator* dense_matcher = new ossimOpenCvDisparityMapGenerator();
    out_disp = dense_matcher->execute(master_mat_8U, slave_mat_warp);

   // ogni disp map deve essere ruotata, convertita a CV_64F, divisa per 16 bit, resa metrica tramite il conv fact

    // Rotation for along-track OPTICAL images
    //********* To be commented for SAR images *********
    //cv::transpose(out_disp, out_disp);
    //cv::flip(out_disp, out_disp, 0);
    //********* To be commented for SAR images *********

    out_disp.convertTo(out_disp, CV_64F);
    out_disp = ((out_disp/16.0)) / ConversionFactor; //quando divido per il fattore di conversione le rendo metriche

    // Nel vettore globale di cv::Mat immagazzino tutte le mappe di disparità che genero ad ogni ciclo
    fusedDisp_array.push_back(out_disp);
    //null_disp_threshold = (dense_matcher->minimumDisp)+0.5;*/

    return true;
}

struct ResponseComparator
{
    bool operator() (const cv::KeyPoint& a, const cv::KeyPoint& b)
    {
        return std::abs(a.response) > std::abs(b.response);
    }
};

void ossimOpenCvTPgenerator::TPgen()
{
    cv::Ptr<cv::CLAHE> filtro = cv::createCLAHE(8.0);
    filtro->apply(master_mat, master_mat);
    filtro->apply(slave_mat, slave_mat);

    

    int maxTotalKeypoints = 500;
    int gridRows_ = 1;
    int gridCols_ = 1;
    int maxPerCell_ = maxTotalKeypoints / (gridRows_ * gridCols_);



	keypoints1.reserve(maxTotalKeypoints);
	keypoints2.reserve(maxTotalKeypoints);
    // Computing detector

    //cv::OrbFeatureDetector detector(30000);//, 2.0f,8, 151, 0, 2, cv::ORB::HARRIS_SCORE, 151 ); // edgeThreshold = 150, patchSize = 150);
    //detector.detect(master_mat, keypoints1);
    //detector.detect(slave_mat, keypoints2);

    cv::Ptr<cv::FeatureDetector> m_detector;
    //cv::Ptr<cv::OrbFeatureDetector> detector = cv::FeatureDetector::create("ORB");
    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    //detector->setMaxFeatures(500);
    detector->setMaxFeatures(maxPerCell_);

    //m_detector = new cv::GridAdaptedFeatureDetector (detector, 500, 5, 5 );
    // riga di codice problematica perché GridAdaptedFeatureDetector non esiste più in OpenCV 3
    m_detector = detector;
    //m_detector->detect(master_mat, keypoints1);
    //m_detector->detect(slave_mat, keypoints2);


    // for loop over the grid containing the sub images
    for(int i = 0; i < gridRows_* gridCols_; i++)
    {

    	int celly = i / gridCols_;
        int cellx = i - celly * gridCols_;

        // sort of slicing as python
        cv::Range row_range_master((celly*master_mat.rows)/gridRows_, ((celly+1)*master_mat.rows)/gridRows_);
        cv::Range col_range_master((cellx*master_mat.cols)/gridCols_, ((cellx+1)*master_mat.cols)/gridCols_);

        cv::Range row_range_slave((celly*slave_mat.rows)/gridRows_, ((celly+1)*slave_mat.rows)/gridRows_);
        cv::Range col_range_slave((cellx*slave_mat.cols)/gridCols_, ((cellx+1)*slave_mat.cols)/gridCols_);
    
        cv::Mat sub_image_master = master_mat(row_range_master, col_range_master);
        cv::Mat sub_image_slave = slave_mat(row_range_slave, col_range_slave);

        cv::Mat sub_mask_master;
            if (!mask_master.empty()) sub_mask_master = mask_master(row_range_master, col_range_master);
        cv::Mat sub_mask_slave;
            if (!mask_slave.empty()) sub_mask_slave = mask_slave(row_range_slave, col_range_slave);

        std::vector<cv::KeyPoint> sub_keypoints1;
        //alloco lo spazio di memoria
        sub_keypoints1.reserve(maxPerCell_);
        std::vector<cv::KeyPoint> sub_keypoints2;
        //alloco lo spazio di memoria
        sub_keypoints2.reserve(maxPerCell_);
        //searching for Key Points in both sub-images
        m_detector->detect(sub_image_master, sub_keypoints1, sub_mask_master);
   		m_detector->detect(sub_image_slave, sub_keypoints2, sub_mask_slave);

        if(sub_keypoints1.size() > maxPerCell_)
            {
                std::vector<cv::KeyPoint>::iterator nth = sub_keypoints1.begin() + maxPerCell_;
                std::nth_element(sub_keypoints1.begin(), nth, sub_keypoints1.end(), ResponseComparator());
                sub_keypoints1.erase( nth, sub_keypoints1.end() );
            }

        if(sub_keypoints2.size() > maxPerCell_)
            {
                std::vector<cv::KeyPoint>::iterator nth = sub_keypoints2.begin() + maxPerCell_;
                std::nth_element(sub_keypoints2.begin(), nth, sub_keypoints2.end(), ResponseComparator());
                sub_keypoints2.erase( nth, sub_keypoints2.end() );
            }

   		//moving from single cell to overall image 
   		std::vector<cv::KeyPoint>::iterator it = sub_keypoints1.begin(),
                                                end = sub_keypoints1.end();
            for( ; it != end; ++it )
            {
                it->pt.x += col_range_master.start;
                it->pt.y += row_range_master.start;
            }

   		std::vector<cv::KeyPoint>::iterator it2 = sub_keypoints2.begin(),
                                                end2 = sub_keypoints2.end();
            for( ; it2 != end2; ++it2 )
            {
                it2->pt.x += col_range_slave.start;
                it2->pt.y += row_range_slave.start;
            }   
        //
        keypoints1.insert( keypoints1.end(), sub_keypoints1.begin(), sub_keypoints1.end() );
        keypoints2.insert( keypoints2.end(), sub_keypoints2.begin(), sub_keypoints2.end() ); 

        cout << "grid cell " << cellx << ' ' << celly <<endl;     
    }
    
    

    for (int i = 0; i < maxTotalKeypoints; i++)
    {
   		double xKP1 = keypoints1[i].pt.x;
    	double yKP1 = keypoints1[i].pt.y;
    	double xKP2 = keypoints2[i].pt.x;
    	double yKP2 = keypoints2[i].pt.y;
    	cout << " xKP1 "  << xKP1 << " yKP1 " << yKP1 << " xKP2 " << xKP2 << " yKP2 " << yKP2 << endl;
	}


    cerr << "Features found = " << keypoints1.size() << " master " << keypoints2.size() << " slave " << endl;

    // Computing descriptors
    //cv::BriefDescriptorExtractor extractor;
    cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> extractor =cv::xfeatures2d::BriefDescriptorExtractor::create ( );

    cv::Mat descriptors1, descriptors2;
    /*extractor.compute(master_mat, keypoints1, descriptors1);
    extractor.compute(slave_mat, keypoints2, descriptors2);*/
    extractor->compute(master_mat, keypoints1, descriptors1);
    extractor->compute(slave_mat, keypoints2, descriptors2);


    // Matching descriptors
    cv::BFMatcher matcher(cv::NORM_L2);
    vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);	

    cerr << "matches " << matches.size() << endl;

    // Calculation of max and min distances between keypoints 
    double max_dist = matches[0].distance; double min_dist = matches[0].distance;
    cout << "max dist" << max_dist << endl;
    cout << "min dist" << min_dist << endl;

    for( int i = 1; i < descriptors1.rows; i++ )
    {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    cout << "Min dist between keypoints = " << min_dist << endl;
    cout << "Max dist between keypoints = " << max_dist << endl;

    // Selection of the better 2% descriptors
    int N_TOT = descriptors1.rows;
    int N_GOOD = 0, N_ITER = 0;

    double good_dist = (max_dist+min_dist)/2.0;
    double per = 100;
    cerr << "good_dist 2 " << good_dist << endl;
    while (fabs(per-0.98) > 0.001 && N_ITER <= 200)
    {
        for( int i = 0; i < descriptors1.rows; i++ )
        {
            if(matches[i].distance <= good_dist) N_GOOD++;
        }

        per = (double)N_GOOD/(double)N_TOT;

        if(per >= 0.98)
        {
            max_dist = good_dist;
        }
        else
        {
            min_dist = good_dist;
        }

        good_dist = (max_dist+min_dist)/2.0;

        //cout<< per << " " << min_dist << " " << max_dist << " "<< good_dist <<endl;

        N_ITER++;
        N_GOOD = 0;
    }
    cerr << "good_dist 3 " << good_dist << endl;
    for( int i = 0; i < descriptors1.rows; i++ )
    {
        if(matches[i].distance <= good_dist)
        {
            // Parallax computation
            double px = keypoints1[i].pt.x - keypoints2[matches[i].trainIdx].pt.x;
            double py = keypoints1[i].pt.y - keypoints2[matches[i].trainIdx].pt.y;	

            if(fabs(py) <  20)
            {
                good_matches.push_back(matches[i]);

                //acc_x(px);
                //acc_y(py);

                //cout << i << " " << px << " " << " " << py << " "<<endl;	
            }
        }
    }
    cout << "good " << (double)good_matches.size() << " matches " << (double)matches.size() << endl;
    cout << "% points found = " << ((double)good_matches.size()/(double)matches.size())*100.0 << endl;
    cout << endl << "Points found before the 3 sigma test = " << (double)good_matches.size() <<endl << endl; 

    // 3 sigma test
    cout << "3 SIGMA TEST " << endl;
    int control = 0;
    int num_iter = 0;

    do
    {
        num_iter ++;
        cout << "Iteration n = " << num_iter << endl;
        control = 0;

        cv::Mat parallax = cv::Mat::zeros(good_matches.size(), 1, CV_64F);
        for(size_t i = 0; i < good_matches.size(); i++)
        {
            parallax.at<double>(i,0) = keypoints1[good_matches[i].queryIdx].pt.y - keypoints2[good_matches[i].trainIdx].pt.y; 	
        }
        cv::Scalar mean_parallax, stDev_parallax;
        cv::meanStdDev(parallax, mean_parallax, stDev_parallax);

        double dev_y = stDev_parallax.val[0]; 	
        double mean_diff_y = mean_parallax.val[0]; 

        cout << "dev_y = " << dev_y << endl
             << "mean_diff_y = " << mean_diff_y << endl;


        vector<cv::DMatch > good_matches_corr;

        // Get the keypoints from the good_matches
        for (size_t i = 0; i < good_matches.size(); i++)
        {
            double py = keypoints1[good_matches[i].queryIdx].pt.y - keypoints2[good_matches[i].trainIdx].pt.y;

            if (py< 3*dev_y+mean_diff_y && py > -3*dev_y+mean_diff_y)        
            {
                good_matches_corr.push_back(good_matches[i]);
            }
            else //find outlier 
            {
                control = 10;
            }
        }
        good_matches = good_matches_corr;
    }
    while(control !=0);
    cout << endl << "Good points found after the 3 sigma test = " << (double)good_matches.size() <<endl << endl; 
}

void ossimOpenCvTPgenerator::TPdraw()
{
    /*cv::Mat filt_master, filt_slave;
    cv::Ptr<cv::CLAHE> filtro = cv::createCLAHE(3.0);
    filtro->apply(master_mat, filt_master);
    filtro->apply(slave_mat, filt_slave);*/

    // Drawing the results
    cv::Mat img_matches;
    cv::drawMatches(master_mat, keypoints1, slave_mat, keypoints2,
               good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
               vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

    cv::resize(img_matches, img_matches, cv::Size(), 1.0/1.0, 1.0/1.0, cv::INTER_AREA);

    cv::namedWindow("TP matched", cv::WINDOW_NORMAL );
    cv::imshow("TP matched", img_matches );

    cv::waitKey(0);
}

bool ossimOpenCvTPgenerator::TPwarp()
{
    std::vector<cv::Point2f> aff_match1, aff_match2;
    // Get the keypoints from the good_matches
    for (unsigned int i = 0; i < good_matches.size(); ++i)
    {
        cv::Point2f pt1 = keypoints1[good_matches[i].queryIdx].pt;
        cv::Point2f pt2 = keypoints2[good_matches[i].trainIdx].pt;
        aff_match1.push_back(pt1);
        aff_match2.push_back(pt2);
        //printf("%3d pt1: (%.2f, %.2f) pt2: (%.2f, %.2f)\n", i, pt1.x, pt1.y, pt2.x, pt2.y);
    }

    // Estimate quasi-epipolar transformation model 
    cv::Mat rot_matrix = estRT(aff_match2, aff_match1);
    cout << "matrice di rotazione " << rot_matrix << endl;

    // Set the destination image the same type and size as source
    cv::Mat warp_dst = cv::Mat::zeros(master_mat.rows, master_mat.cols, master_mat.type());
    //cv::Mat warp_dst_16bit = cv::Mat::zeros(slave_16bit.rows, slave_16bit.cols, slave_16bit.type());

    //cout << "Warp pre" << warp_dst << endl;

    cv::warpAffine(slave_mat, warp_dst, rot_matrix, warp_dst.size());
    // perchè rot_matrix non ha tx nulla?
    //cv::warpAffine(slave_16bit, warp_dst_16bit, rot_matrix, warp_dst.size());

    //cout << "Warp post" << warp_dst << endl;

    //cv::namedWindow("Master image", cv::WINDOW_NORMAL);
    //cv::imshow("Master image", master_mat );
    cv::imwrite("Master_8bit.tif",  master_mat);

    //cv::namedWindow("Warped image", cv::WINDOW_NORMAL);
    //cv::imshow("Warped image", warp_dst );
    cv::imwrite("Slave_8bit.tif",  warp_dst);

    //slave_mat_warp = warp_dst;
    slave_mat_warp = slave_mat; // for SAR or images with RPC bias-corrected
    //cv::waitKey(0);

    return true;
}


cv::Mat ossimOpenCvTPgenerator::getWarpedImage()
{
    return slave_mat_warp;
}


cv::Mat ossimOpenCvTPgenerator::estRT(std::vector<cv::Point2f> master, std::vector<cv::Point2f> slave)
{
    size_t m = master.size();

    if ( master.size() != slave.size() )
    {
        throw 0;
    }

    // Computing barycentric coordinates

    cv::Scalar mean_master_x , mean_master_y, mean_slave_x, mean_slave_y, mean_shift_x, mean_shift_y;
    cv::Scalar stDev_master_x, stDev_master_y, stDev_slave_x, stDev_slave_y, stDev_shift_x, stDev_shift_y;

    cv::Mat components_matrix = cv::Mat::zeros(m,6, CV_64F);
    for(size_t i = 0; i < m; i++)
    {
        components_matrix.at<double>(i,0) = master[i].x;
        components_matrix.at<double>(i,1) = master[i].y;
        components_matrix.at<double>(i,2) = slave[i].x;
        components_matrix.at<double>(i,3) = slave[i].y;
        components_matrix.at<double>(i,4) = master[i].x - slave[i].x;
        components_matrix.at<double>(i,5) = master[i].y - slave[i].y;
    }
    cv::meanStdDev(components_matrix.col(0), mean_master_x, stDev_master_x);
    cv::meanStdDev(components_matrix.col(1), mean_master_y, stDev_master_y);
    cv::meanStdDev(components_matrix.col(2), mean_slave_x, stDev_slave_x);
    cv::meanStdDev(components_matrix.col(3), mean_slave_y, stDev_slave_y);
    cv::meanStdDev(components_matrix.col(4), mean_shift_x, stDev_shift_x);
    cv::meanStdDev(components_matrix.col(5), mean_shift_y, stDev_shift_y);

    master_x = mean_master_x.val[0];
    master_y = mean_master_y.val[0];
    slave_x = mean_slave_x.val[0];
    slave_y	= mean_slave_y.val[0];


    double StDevShiftX = stDev_shift_x.val[0];
    double StDevShiftY = stDev_shift_y.val[0];


    cout << "Mean_x_master = " << master_x << endl
         << "Mean_y_master = " << master_y << endl
         << "Mean_x_slave = "  << slave_x  << endl
         << "Mean_y_slave = "  << slave_y  << endl
         << "Shift in x = "	<< master_x - slave_x << "\tpixel" << endl
         << "Shift in y = "	<< master_y - slave_y << "\tpixel" <<endl << endl
         << "St.dev. shift x = " <<	StDevShiftX << endl
         << "St.dev. shift y = " <<	StDevShiftY << endl << endl;

    std::vector<cv::Point2f> bar_master, bar_slave;

    for (size_t i = 0; i < m; i++)
    {
        cv::Point2f pt1;
        cv::Point2f pt2;

        pt1.x = master[i].x - master_x;
        pt1.y = master[i].y - master_y;

        //pt2.x = slave[i].x - master_x;
        //pt2.y = slave[i].y - master_y;

        pt2.x = slave[i].x - slave_x;
        pt2.y = slave[i].y - slave_y;

        //cout << pt1.x << "\t" << pt1.y << "\t" << pt2.x << "\t" << pt2.y<< "\t" << pt1.y-pt2.y<< endl;

        bar_master.push_back(pt1);
        bar_slave.push_back(pt2);
    }

    /// ***rigorous model start***
    cv::Mat x_approx = cv::Mat::zeros (2+m,1,6);
    cv::Mat result = cv::Mat::zeros (2+m, 1, 6);
    cv::Mat A = cv::Mat::zeros(2*m,2+m,6);
    cv::Mat B = cv::Mat::zeros(2*m,1,6);

    cv::Mat trX;
    double disp_median = master_x-slave_x;

    for (int j= 0; j <3; j++)
    {
        for (size_t i=0; i < m ; i++)
        {
            A.at<double>(2*i,0) = bar_slave[i].y;
            A.at<double>(2*i,1) = 0.0;
            A.at<double>(2*i,2+i) = 1.0;

            A.at<double>(2*i+1,0) = -bar_slave[i].x- disp_median;
            A.at<double>(2*i+1,1) = 1.0;
            A.at<double>(2*i+1,2+i) = 0.0;

            B.at<double>(2*i,0)   = bar_master[i].x - cos(x_approx.at<double>(0,0))*(bar_slave[i].x  +disp_median+ x_approx.at<double>(2+i,0))
                                                     - sin(x_approx.at<double>(0,0))*bar_slave[i].y;
            B.at<double>(2*i+1,0) = bar_master[i].y + sin(x_approx.at<double>(0,0))*(bar_slave[i].x  +disp_median+ x_approx.at<double>(2+i,0))
                                                     - cos(x_approx.at<double>(0,0))*bar_slave[i].y - x_approx.at<double>(1,0);
        }

        cv::solve(A, B, result, cv::DECOMP_SVD);
        x_approx = x_approx+result;

        cv::Mat trX;
        cv::transpose(result, trX);
        //cout << "Matrice risultati\n" << x_approx << endl;
        //cout << "Result matrix "<< endl;
        //cout << trX << endl << endl;

        cv::transpose(x_approx, trX);

        //cout << "X approx matrix iteration " << j << endl;
        //cout << trX << endl << endl;
    }

    //cout << "Difference " << endl;
    //cout << A*x_approx-B << endl;

    //cout << A << endl;
    //cout << B << endl;

    trX = A*x_approx-B;

    /*for(size_t i=0; i < m ; i++)
    {
        cout << master[i].y <<"\t" << trX.at<double>(2*i+1,0) <<endl;
    }*/

    // rotation is applied in the TPs barycenter
    //cv::Point2f pt(master_x , master_y);
    cv::Point2f pt(slave_x , slave_y);
    cv::Mat r = cv::getRotationMatrix2D(pt, -x_approx.at<double>(0,0)*180.0/3.141516, 1.0);
    r.at<double>(1,2) += x_approx.at<double>(1,0) - master_y + slave_y;

    /// ***rigorous model end***

    /// ***linear model start***
    /*cv::Mat result = cv::Mat::zeros (2, 1, 6);
    cv::Mat A = cv::Mat::zeros(m,2,6);
    cv::Mat B = cv::Mat::zeros(m,1,6);

    for (size_t i=0; i<m ; i++)
        {
            A.at<double>(i,0) = bar_slave[i].y;
            A.at<double>(i,1) = 1.0;

            B.at<double>(i,0) = bar_master[i].y;
        }

    cv::solve(A, B, result, cv::DECOMP_SVD);
    cv::Mat trX;
    cv::transpose(result, trX);
    cout << "Result matrix "<< endl;
    cout << trX << endl << endl;
    cout << "Difference " << endl;
    cout << A*result-B << endl;

    cv::Mat r = cv::Mat::zeros (2, 3, 6);
    r.at<double>(0,0) = 1.0;
    r.at<double>(0,1) = 0.0;
    r.at<double>(0,2) = 0.0;
    r.at<double>(1,0) = 0.0;
    r.at<double>(1,1) = result.at<double>(0,0);
    r.at<double>(1,2) = result.at<double>(1,0) - master_y + slave_y;*/

    /// ***linear model end***
    //cout << "Matrice r" << r << endl;
    return r;
}