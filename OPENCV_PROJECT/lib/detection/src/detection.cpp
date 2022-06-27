//detection.cpp

#ifndef _DET_
#define _DET_
#include "detection.h"
#endif 

//CONSTRUCTORS

//default one
HandDetector::HandDetector(){

    size = 416;
    normalize = 1/255.0f;

    nmaxima_thresh = 0.4f;
    conf_threshold = 0.6f;

    yolo_net = readNetFromDarknet("./../../yolo_files/yolov4.cfg", "./../../yolo_files/yolov4-obj_3000.weights");

    model = DetectionModel(yolo_net);
    model.setInputParams(normalize, cv::Size(size, size), cv::Scalar(), true);

}

//overloads

HandDetector::HandDetector(const std::string& cfg_path, const std::string& weights_path){

    size = 416;
    normalize = 1/255.0f;

    nmaxima_thresh = 0.4f;
    conf_threshold = 0.6f;

    yolo_net = readNetFromDarknet(cfg_path, weights_path);

    model = DetectionModel(yolo_net);
    model.setInputParams(normalize, cv::Size(size, size), cv::Scalar(), true);

}

HandDetector::HandDetector(const std::string& cfg_path, const std::string& weights_path, int size, float conf_threshold, float nmaxima_thresh) : size(size), conf_threshold(conf_threshold), nmaxima_thresh(nmaxima_thresh){
  
    normalize = 1/255.0f;

    yolo_net = readNetFromDarknet(cfg_path, weights_path);

    model = DetectionModel(yolo_net);
    model.setInputParams(normalize, cv::Size(size, size), cv::Scalar(), true);

}


HandDetector::HandDetector(float conf_threshold, float nmaxima_thresh) : conf_threshold(conf_threshold), nmaxima_thresh(nmaxima_thresh){
  
    normalize = 1/255.0f;
    size = 416;
    yolo_net = readNetFromDarknet("./../../yolo_files/yolov4.cfg", "./../../yolo_files/yolov4-obj_3000.weights");

    model = DetectionModel(yolo_net);
    model.setInputParams(normalize, cv::Size(size, size), cv::Scalar(), true);

}






//PUBLIC MEMBERS
void HandDetector::detect_hands(const cv::Mat& image, cv::Mat& output){

    std::vector<int> idx;
    std::vector<float> scores;
    std::vector<cv::Rect> boxes;
    model.detect(image, idx, scores, boxes, conf_threshold, nmaxima_thresh);

    draw_bbox(image, output, boxes);

}

void  HandDetector::detect_hands(const cv::Mat& image, std::vector<float>& scores,std::vector<cv::Rect>& boxes){

    std::vector<int> idx;
    
    model.detect(image, idx, scores, boxes, conf_threshold, nmaxima_thresh);
    std::cout<<boxes.size()<<std::endl;

}


//PRIVATE MEMBERS
void HandDetector::draw_bbox(const cv::Mat& image, cv::Mat& output, std::vector<cv::Rect>& boxes){

    output = image.clone();

    for (int i = 0; i < boxes.size(); i++) {
        cv::rectangle(output, boxes[i], cv::Scalar(0, 255, 0), 2);
        
    }

}


void detection_print(){

    std::cout<<"detection ready"<<std::endl;
}

