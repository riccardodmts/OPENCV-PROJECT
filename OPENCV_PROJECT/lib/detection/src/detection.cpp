//detection.cpp

#ifndef _DET_
#define _DET_
#include "detection.h"
#endif 


//CONSTRUCTORS

//default one

HandDetector::HandDetector(){

    net =  readNetFromDarknet("./../../../yolo_files/yolov4.cfg", "./../../../yolo_files/yolov4-obj_3000.weights");
    
    width = 416;
    height = 416;
    conf_th = 0.5f;
    nms_th = 0.4f;
    normalize = 1.0f/255.0f;

}

HandDetector::HandDetector(float conf_th, float nms_th, int size) : width(size), height(size), conf_th(conf_th), nms_th(nms_th){

    net =  readNetFromDarknet("./../../yolo_files/yolov4.cfg", "./../../yolo_files/yolov4-obj_3000.weights");

    normalize = 1.0f/255.0f;

}

HandDetector::HandDetector(const std::string& cfg_path, const std::string& weights_path){

    net =  readNetFromDarknet(cfg_path, weights_path);
    
    width = 416;
    height = 416;
    conf_th = 0.5f;
    nms_th = 0.4f;
    normalize = 1.0f/255.0f;

}

HandDetector::HandDetector(float conf_th, float nms_th, int size, const std::string& cfg_path, const std::string& weights_path) : width(size), height(size), conf_th(conf_th), nms_th(nms_th) {

    net =  readNetFromDarknet(cfg_path, weights_path);

    normalize = 1.0f/255.0f;

}


//PUBLIC METHODS

void HandDetector::detect_hands(const cv::Mat& image, std::vector<float>& confs, std::vector<cv::Rect>& bboxes){

    cv::Mat blob;
    blobFromImage(image, blob, normalize, cv::Size(width, height), cv::Scalar(0,0,0), true, false);

    net.setInput(blob);

    std::vector<cv::Mat> output;
    net.forward(output, net.getUnconnectedOutLayersNames());

    process_results(output, confs, bboxes, image.rows, image.cols);

}


void HandDetector::draw_bboxes(const cv::Mat& image, cv::Mat& output, std::vector<cv::Rect>& bboxes){

    output = image.clone();

    for(size_t i = 0; i < bboxes.size(); i++){

        cv::rectangle(output, bboxes[i], cv::Scalar(0, 255, 0), 2);

    }
}

void HandDetector::detect_hands(const cv::Mat& image, cv::Mat& output){

    std::vector<float> confs;
    std::vector<cv::Rect> bboxes;

    detect_hands(image, confs, bboxes);
    
    draw_bboxes(image, output, bboxes);
}


//PRIVATE METHODS

void HandDetector::process_results(const std::vector<cv::Mat>& output, std::vector<float> confs, std::vector<cv::Rect>& bboxes, int rows, int cols){

    std::vector<int> idxs;

    std::vector<float> temp_conf;
    std::vector<cv::Rect> temp_bboxes;

    

    //for each yolo layer
    for (size_t i = 0; i < output.size(); i++){

        float * data = (float*)output[i].data;

        for(int j = 0; j < output[i].rows; j++, data += output[i].cols){

            cv::Mat scores = output[i].row(j).colRange(5, output[i].cols);

            cv::Point point;
            double conf;

            minMaxLoc(scores, 0, &conf, 0, &point);

            if(conf > conf_th){

                //yolo returns normalized coordinates/dimensions

                int cx = (int)(data[0] * cols);
                int cy = (int)(data[1] * rows);
                int w  = (int)(data[2] * cols);
                int h  = (int)(data[3] * rows);

                int left = cx - w/2;
                int top  = cy - h/2;

                
                temp_conf.push_back((float)conf);
                temp_bboxes.push_back(cv::Rect(left, top, w, h));

                
            }

        }


    }

   
    NMSBoxes(temp_bboxes, temp_conf, conf_th, nms_th, idxs);

    for(size_t i = 0; i < idxs.size(); i++){
        int idx = idxs[i];
        cv::Rect temp = temp_bboxes[idx];
        resize_bbox(temp, rows, cols);
        confs.push_back(temp_conf[idx]);
        bboxes.push_back(temp);
    }


}

void HandDetector::resize_bbox(cv::Rect& bbox, int rows, int cols){
    
    if(bbox.x < 0) bbox.x = 0;
    if(bbox.y < 0) bbox.y = 0;

    if(bbox.x + bbox.width > cols) bbox.width = cols - bbox.x;
    if(bbox.y + bbox.height > rows) bbox.height = rows - bbox.y;

}
