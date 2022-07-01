//skin_segmentation.h

#ifndef SKIN_SEGMENTATION_H
#define SKIN_SEGMENTATION_H

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#endif

//implementation of the k-means clustering algorithm
cv::Mat K_Means(cv::Mat Input, int K, cv::Mat &RGB_centers);

//compute the region of the image that can be considered as skin
cv::Mat get_skin(cv::Mat Input_image);