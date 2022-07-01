//MAIN APP

#ifndef _MAIN_  //standard libraries
#define _MAIN_
#include <iostream>
#include <opencv2/opencv.hpp>
#endif

#ifndef _SEGMENT_  //graph segmentation class
#define _SEGMENT_
#include "get_segmentation.h"
#endif

#ifndef SKIN_SEG //skin detection algorithm
#define SKIN_SEG
#include "skin_segmentation.h"
#endif


#ifndef _DET_ //detection class (CNN based)
#define _DET_
#include "detection.h"
#endif 


int main ( int argc, char** argv )
{
    //////////////////////  SUMMARY (TO DO)   //////////////////
    /*
    -CLASSIFY
    -GET ROI (BOUNDING BOX) FROM CLASSIFIER
    -APPLY SKIN DETECTION & SEGMENT
    -APPLY MASK (BOUNDING BOX) TO SEGMENTED IMAGE
    -FIND NUMBER OF CLASSES
    -FOR EACH CLASS SUBSTITUTE ORIGINAL PIECE OF IMAGE
    -RE-RUN CLASSIFIER : IF HAND TAKE THAT REGION AS CORRECTLY SEGMENTED OTHERWISE TRY FOR NEW CLASS
    */

    HandDetector detector;
    HandSegmentor segment(&detector);


    cv::Mat prova = cv::imread("./../../../29.jpg");
    cv::imwrite("./../../../pr1.ppm", prova);

    segment.get_segmentation();

    cv::Mat segmented = cv::imread("./../../../segmentated/prova.ppm");
    cv::imshow("segmented", segmented);

    segment.test();


    cv::waitKey(0);
    return 0;
}