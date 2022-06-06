//main program


#include <stdio.h>
#include <iostream>

#include "segmentation.h"
#include "detection.h"
#include "classifier.h"

#include <opencv2/opencv.hpp>

int main ( int argc, char** argv )
{
    //image loading
    cv::Mat Input_Image = cv::imread("./../rgb/29.jpg");  
    if (!Input_Image.data)
    {
        printf("No image data \n");
        return -1;
    }
    // Show the source image
    cv::imshow("Source Image", Input_Image);

    cv::waitKey(0);
    return 0;
}
