//main program

#include <iostream>

#include "skin_segmentation.h"
#include "detection.h"
#include "classifier.h"
#include "get_segmentation.h"

int main ( int argc, char** argv )
{


    //////////////////////  SUMMURY (TO DO)   //////////////////
    /*
    -CLASSIFY
    -GET ROI (BOUNDING BOX) FROM CLASSIFIER
    -APPLY SKIN DETECTION & SEGMENT
    -APPLY MASK (BOUNDING BOX) TO SEGMENTED IMAGE
    -FIND NUMBER OF CLASSES
    -FOR EACH CLASS SUBSTITUTE ORIGINAL PIECE OF IMAGE
    -RE-RUN CLASSIFIER : IF HAND TAKE THAT REGION AS CORRECTLY SEGMENTED OTHERWISE TRY FOR NEW CLASS
    */

    float sigma = 0.5;
    float k = 500;
    int min_size = 20;

    cv::Mat prova = cv::imread("./../../prov.jpg");
    cv::imwrite("./../../pr1.ppm", prova);

    const char *input_path = "./../../pr1.ppm";
    const char *output_path = "./../../segmentated/prova.ppm";
        
    get_segmentation(sigma, k, min_size, input_path, output_path);

    cv::Mat segmented = cv::imread("./../../segmentated/prova.ppm");
    cv::imshow("segmented", segmented);

    cv::waitKey(0);
    return 0;
}

