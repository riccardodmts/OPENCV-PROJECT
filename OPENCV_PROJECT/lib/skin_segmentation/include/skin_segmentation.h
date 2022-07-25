//skin_segmentation.h

#ifndef SKIN_SEGMENTATION_H
#define SKIN_SEGMENTATION_H

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#endif

//implementation of the k-means clustering algorithm: given an image, the number of clusters (k) and the centroids initial values it will 
//segment the image based on k-means algorithm and returns a cv::Mat output image
cv::Mat K_Means(cv::Mat Input, int K, cv::Mat &RGB_centers);

//compute the region of the image that can be considered as skin. Given an input image it checks the intensity values of each pixel to 
// understand if it can be considered as skin or not, based on some thresholds
cv::Mat get_skin(cv::Mat Input_image);
