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

        Net net;
        float nms_th, conf_th;
        int width, height;

        float normalize;

    public:

        HandDetector();

        //change params
        HandDetector(float conf_th, float nms_th, int size);

        //specify path for .cfg and .weights
        HandDetector(const std::string& cfg_path, const std::string& weights_path);

        //change params and specify path for .cfg and .weights
        HandDetector(float conf_th, float nms_th, int size, const std::string& cfg_path, const std::string& weights_path);
        
        //this method returns the bounding boxes and the corresponding confidence levels
        //it calls HandDetector::process_results
        void detect_hands(const cv::Mat& image, std::vector<float>& confs, std::vector<cv::Rect>& bboxes);

        //this method detects hands and draws the bounding boxes on output (output will be an hard copy of image)
        //to draw, it calls HandDetector::draw_bboxes , while for the detection it calls the previous HandDetector::detect_hands
        void detect_hands(const cv::Mat& image, cv::Mat& output);

        //draws the bounding boxes on output (output will be an hard copy of image)
        void draw_bboxes(const cv::Mat& image, cv::Mat& output, std::vector<cv::Rect>& bboxes);

    private:

        //this method processes the output of the CNN
        void process_results(const std::vector<cv::Mat>& output, std::vector<float> confs, std::vector<cv::Rect>& bboxes, int rows, int cols);

};
