//detection.h
//author: Riccardo De Monte
//hours spent: 2



#ifndef _STD_
#define _STD_
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <math.h>
#endif

using namespace cv::ml;
using namespace cv::dnn;

// HandDetector class for hands detection: based on yolov4 trained on darknet
//two files needed: .cfg and .weights

class HandDetector{

    private:

        const std::string class_name = "Hand";

        Net yolo_net;

        DetectionModel model;

        std::vector<Net> nets; //not used
        std::vector<DetectionModel> models; //not used

        int size;
        float normalize;

        float nmaxima_thresh;
        float conf_threshold;

    public:

        //CONSTRUCTORS

        HandDetector();
        HandDetector(float conf_threshold, float nmaxima_thresh);
        HandDetector(const std::string& cfg_path, const std::string& weigths_path);
        HandDetector(const std::string& cfg_path, const std::string& weigths_path, int size, float conf_threshold, float nmaxima_thresh);

        //PUBLIC METHODS

        //detect hands: it returns the boxes and the corresponding confidences
        void detect_hands(const cv::Mat& image, std::vector<float>& scores, std::vector<cv::Rect>& boxes);

        //detect hands: it returns (output) the image with the boxes drawn
        void detect_hands(const cv::Mat& image, cv::Mat & output);
    
    private:

        //PRIVATE METHODS
        //draw bounding boxes, provided the results of the hads detection (the boxes)
        void draw_bbox(const cv::Mat& image, cv::Mat & output, std::vector<cv::Rect>& boxes);


};

//
void detection_print();




