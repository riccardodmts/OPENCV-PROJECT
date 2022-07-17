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


int main ( int argc, char* argv[] )
{

    HandDetector detector;
    HandSegmentor segment(&detector);


    std::string str_path = "../../test/rgb/" + std::string(argv[1]);
    char* img_path = &str_path[0];


    cv::Mat prova = cv::imread(img_path);

    std::vector<cv::Rect> boxes;
    std::vector<float> conf;
    detector.detect_hands(prova, conf, boxes);


    cv::Mat maskk = segment.final_mask(img_path, boxes);


    Evaluation ev(img_path, maskk, boxes);
    cv::Mat iou_img;
    cv::Mat accuracy_img = ev.PixelAccuracy();


    if(argc < 3)
      iou_img = ev.IoU(0); //mode with only the detected bounding box and the IoU result on console

    else
      iou_img = ev.IoU(1); //mode with the IoU result on the image

    cv::imshow("Original Image", prova);
    cv::imshow("Accuracy Image", accuracy_img);
    cv::imshow("Bounding Boxes", iou_img);
    cv::waitKey(0);

    return 0;
}
