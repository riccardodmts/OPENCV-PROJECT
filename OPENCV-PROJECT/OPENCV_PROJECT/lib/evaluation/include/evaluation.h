//evaluation.h

#ifndef EVALUATION_h
#define EVALUATION_h

#include <iostream>
#include <fstream>
#include <math.h>
#include <opencv2/opencv.hpp>

#endif

class Evaluation{

  private:

    cv::Mat image;
    cv::Mat true_mask;
    cv::Mat detected_mask;
    std::vector<cv::Rect> true_bbox;
    std::vector<cv::Rect> detected_bbox;
    bool truth;

    /*
    function that assign to the true bboxes, the corresponding detected bbox

    assign means, to consider the detected bbox that maximize the IoU for that true bbox
    if the detected bbox are less than the true bbox, we copy from the
    detected bbox vector, the bbox that maximize the IoU for the remaining true bboxes.
    */
    void ValidateBoundingBox();

    /*
    function that returns (overwriting the last two parameters) the couple of
    position (true_max, det_max) of the corresponding vectors (true_box, det_bbox)
    that maximize the IoU (among all the possible couple of elements)
    */
    void MaximizeIoU(std::vector<cv::Rect> true_box, std::vector<cv::Rect> det_bbox, int &true_max, int &det_max);

    // return the IoU value of the two bounding boxes
    float IoU(cv::Rect bbox_A, cv::Rect bbox_B);
    cv::Mat Accuracy();

  public:
    Evaluation(std::string img_path, std::string true_mask_path, cv::Mat our_mask, std::string bbox_path, std::vector<cv::Rect> our_bbox);
    Evaluation(std::string img_path, cv::Mat our_mask, std::vector<cv::Rect> our_bbox);
    Evaluation(std::string img_path, cv::Mat our_mask, std::vector<cv::Rect> our_bbox, int aux);

    cv::Mat IoU(int mode);
    cv::Mat PixelAccuracy();

};
