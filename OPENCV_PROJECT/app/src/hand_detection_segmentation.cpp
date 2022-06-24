//main program

#include <stdio.h>
#include <iostream>

#include "skin_segmentation.h"
#include "detection.h"
#include "classifier.h"

//for segmentation//
#include <cstdio>
#include <cstdlib>
#include "image.h"
#include "misc.h"
#include "pnmfile.h"
#include "segment-image.h"

#include <opencv2/opencv.hpp>

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
        
    printf("loading input image.\n");
    image<rgb> *input = loadPPM(input_path);
        
    printf("processing\n");
    int num_ccs; 
    image<rgb> *seg = segment_image(input, sigma, k, min_size, &num_ccs); 
    savePPM(seg, output_path);

    printf("got %d components\n", num_ccs);

    cv::Mat segmented = cv::imread("./../../segmentated/prova.ppm");
    cv::imshow("segmented", segmented);



    cv::waitKey(0);
    return 0;
}
