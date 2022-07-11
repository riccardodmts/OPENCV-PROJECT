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

#ifndef _EV_ //detection class (CNN based)
#define _EV_
#include "evaluation.h"
#endif


int main ( int argc, char** argv )
{

    HandDetector detector;
    HandSegmentor segment(&detector);

    const char* img_path = "../../test/rgb/22.jpg";

    cv::Mat prova = cv::imread(img_path);

    std::vector<cv::Rect> boxes;
    std::vector<float> conf;
    detector.detect_hands(prova, conf, boxes);

    cv::Mat maskk = segment.final_mask(img_path, boxes);
    cv::imshow("mask", maskk);
    cv::waitKey(0);

    Evaluation ev(img_path, maskk, boxes);
    cv::Mat iou_img =  ev.IoU();
    ev.PixelAccuracy();
    cv::imshow("iou", iou_img);
    cv::waitKey(0);
    cv::destroyAllWindows();


    return 0;
}
