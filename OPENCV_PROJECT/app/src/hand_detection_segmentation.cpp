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

    HandDetector detector;
    HandSegmentor segment(&detector);

 
    //DA FARE 
    //-sistemare intersezione
    //-controllare valori per selezione aree (0.5 o 0.6)
    //skin detection non da maschera in b/w
    //verificare funzione b/w


    cv::Mat prova = cv::imread("./../../22.jpg");

    std::vector<cv::Rect> boxes;
    std::vector<float> conf;
    detector.detect_hands(prova, conf, boxes);
    std::vector<cv::Mat> masks;
    segment.final_masks("./../../22.jpg", boxes, masks);

    for(size_t i = 0; i < masks.size(); i++)
    {
        cv::imshow("mask", masks[i]);
        cv::waitKey(0);
    }

    return 0;
}