//get_segmentation.h

#ifndef _DET_
#define _DET_
#include "detection.h"
#include <stdlib.h>
#include "skin_segmentation.h"
#endif

class HandSegmentor{

    private:

    float sigma;
    float k;
    int min_size;

    HandDetector* ptrHandDetector;

    const char *input_path;
    const char *output_path;

    public:

        //CONSTRUCTORS

        HandSegmentor();
        HandSegmentor(HandDetector* ptrHandDetector);
        HandSegmentor(HandDetector* ptrHandDetector, float sigma, float k, int min_size, const char *input_path, const char *output_path);

        //PUBLIC METHODS

        //detect hands: it returns the boxes and the corresponding confidences
        void get_segmentation();

        void test();//per testare, poi da cancellare

        void final_masks(const char* path, std::vector<cv::Rect>& boxes, std::vector<cv::Mat>& masks);//quella che sostituirà test
        void from_skin_to_mask(const cv::Mat& skin_output, cv::Mat& output);

        cv::Mat final_mask(const char* path, std::vector<cv::Rect> boxes);

    //PRIVATE METHODS

    private:

        void get_masks_per_region(cv::Mat& seg_roi, std::vector<cv::Mat>& masks);
        void get_expanded_roi(const cv::Mat& img, const cv::Rect& roi, cv::Rect& new_roi, int max_disp);
        void get_mask_original_size(const cv::Rect& roi, const cv::Rect& new_roi, const cv::Mat& mask, cv::Mat& new_mask);
        void get_mask_original_size(const cv::Rect& roi, const cv::Rect& new_roi, const std::vector<cv::Mat>& masks, std::vector<cv::Mat>& new_masks);
        int test_region(const std::vector<cv::Rect>& boxes, const cv::Rect& original_box);
        void get_idx_of_regions(const cv::Mat& roi_img, const std::vector<cv::Mat>& masks, std::vector<int>& idxs, const cv::Rect& original_box);
        void get_mask_union(const std::vector<cv::Mat>& masks, std::vector<int>& idxs, cv::Mat& final_mask);
        void intersect_masks(const cv::Mat& input_union, const cv::Mat& input_skin, cv::Mat& output);



};


bool is_Greyscale(cv::Mat img, int batch_size); // check if the image is grayscale
