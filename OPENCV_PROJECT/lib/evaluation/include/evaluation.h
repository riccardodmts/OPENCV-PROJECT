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

    cv::Mat image; //source image
    cv::Mat true_mask; //image with the true mask
    cv::Mat detected_mask; //image with the detected mask
    std::vector<cv::Rect> true_bbox; //vector with all the true bbox
    std::vector<cv::Rect> detected_bbox; //vector with all the detected bbox
    bool truth; //auxiliary variable to know if the ground-truth is provided

    /*
    function that assign to the true bboxes, the corresponding detected bbox

    assign means, to consider the detected bbox that maximize the IoU for that true bbox
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

    //method that perform the accuracy for the mode without the ground-truth
    //(called in the PixelAccuracy method when there is not the ground-truth)
    std::vector<cv::Mat> Accuracy();

  public:

    //constuctor that uses all the path
    Evaluation(std::string img_path, std::string true_mask_path, cv::Mat our_mask, std::string bbox_path, std::vector<cv::Rect> our_bbox);

    /*
      from the path of the test image, the true mask path and the real bbox path
      are obtained. Then the first constructor is recalled

      in this case I suppose to have the following folder hierarchy

      test
        |-> rgb (with the test images)
        |-> mask (with all the true mask)
        |-> det (with all the true bbox)

      */
    Evaluation(std::string img_path, cv::Mat our_mask, std::vector<cv::Rect> our_bbox);

    /*
      constructor for the mode without the ground-truth
    */
    Evaluation(std::string img_path, cv::Mat our_mask, std::vector<cv::Rect> our_bbox, int aux);

    /*
      calculate all the IoU value for the real bounding boxes
      accept a parameter which is the mode
      - mode = 0 -> the method return the image with the detected bbox,
                    the IoU results are printed on console
      - mode = 1 -> the method return an image with all the detected bbox, the
                    real bbox and the IoU results for each true bbox

      if the method is used without iniatilizing the ground-truth on the constructor,
      the output is always equal to the mode 0 (there are not the IoU results)
    */
    cv::Mat IoU(int mode);

    /*
      Return a vector of two images: the first one is the image with the segmented hands,
      the second image is the confusion matrix image.
      On console are printed the result of accuracy, precision, recall,
      balance accuracy and F1 score

      if the method is used without iniatilizing the ground-truth on the constructor,
      the output has still the segmented image as first image,
      the second image of the vector will be the mask of the detected hands
    */
    std::vector<cv::Mat> PixelAccuracy();

};

