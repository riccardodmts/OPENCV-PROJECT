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

        //default one
        HandSegmentor();
    
        //include detector
        HandSegmentor(HandDetector* ptrHandDetector);
    
        //include detector  and change parameters
        HandSegmentor(HandDetector* ptrHandDetector, float sigma, float k, int min_size, const char *input_path, const char *output_path);

        //PUBLIC METHODS
    
        //implement the graph-based segmentation
        void get_segmentation();

        //get final masks
        void final_masks(const char* path, std::vector<cv::Rect>& boxes, std::vector<cv::Mat>& masks);
    
        //get a binary mask from skin_segmentation algorithm
        void from_skin_to_mask(const cv::Mat& skin_output, cv::Mat& output);

        cv::Mat final_mask(const char* path, std::vector<cv::Rect> boxes);

    //PRIVATE METHODS

    private:

        //get all the masks found by the detector for each region
        void get_masks_per_region(cv::Mat& seg_roi, std::vector<cv::Mat>& masks);
    
        //expand the roi before applying the detector
        void get_expanded_roi(const cv::Mat& img, const cv::Rect& roi, cv::Rect& new_roi, int max_disp);
    
        //get original roi dimension
        void get_mask_original_size(const cv::Rect& roi, const cv::Rect& new_roi, const cv::Mat& mask, cv::Mat& new_mask);
        void get_mask_original_size(const cv::Rect& roi, const cv::Rect& new_roi, const std::vector<cv::Mat>& masks, std::vector<cv::Mat>& new_masks);
    
        //verify if the detector has found a region associated to a hand (bbox found by the detector has to be at least 0.5 the original bbox or 
        //consider the union of the regions to be at least 0.6 the original bbox)
        int test_region(const std::vector<cv::Rect>& boxes, const cv::Rect& original_box);
    
        //find the regions associated to a hand from the vector of masks found by "get_masks_per_region"
        void get_idx_of_regions(const cv::Mat& roi_img, const std::vector<cv::Mat>& masks, std::vector<int>& idxs, const cv::Rect& original_box);
    
        //if more masks are found for the same region, take the union
        void get_mask_union(const std::vector<cv::Mat>& masks, std::vector<int>& idxs, cv::Mat& final_mask);
    
        void intersect_masks(const cv::Mat& input_union, const cv::Mat& input_skin, cv::Mat& output);
    
        void get_biggest_region(cv::Mat& seg_roi, cv::Mat& mask);
        int count_nozero_pixels(const cv::Mat& mask);




};

//check if the image is grayscale
bool is_Greyscale(cv::Mat img, int batch_size);
