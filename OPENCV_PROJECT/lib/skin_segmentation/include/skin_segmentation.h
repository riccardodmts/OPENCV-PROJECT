//segmentation.h

#ifndef SEGMENTATION_h
#define SEGMENTATION_h

#include <opencv2/opencv.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#endif


cv::Mat K_Means(cv::Mat Input, int K, cv::Mat &RGB_centers);

cv::Mat get_skin(cv::Mat Input_image);
