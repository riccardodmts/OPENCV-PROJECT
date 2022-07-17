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
        //this function creates a vector of masks: a mask for each segmented region
        void get_masks_per_region(cv::Mat& seg_roi, std::vector<cv::Mat>& masks);
    
        //this function returns a new rect in order to expand a roi (it verifies the limits of the image)
        void get_expanded_roi(const cv::Mat& img, const cv::Rect& roi, cv::Rect& new_roi, int max_disp);
    
        //to get the mask with the right dimensions since an expansion is applied before
        void get_mask_original_size(const cv::Rect& roi, const cv::Rect& new_roi, const cv::Mat& mask, cv::Mat& new_mask);
        
        //it calls the method above for each mask in a vector
        void get_mask_original_size(const cv::Rect& roi, const cv::Rect& new_roi, const std::vector<cv::Mat>& masks, std::vector<cv::Mat>& new_masks);
    
        //test if a regions contains or not an hand (0.5 and 0.6 tests are applied)
        int test_region(const std::vector<cv::Rect>& boxes, const cv::Rect& original_box);
        
        //it computes the indexes of those regions that satisfies the test (oterating over different masks, it calls test_region for each mask)
        void get_idx_of_regions(const cv::Mat& roi_img, const std::vector<cv::Mat>& masks, std::vector<int>& idxs, const cv::Rect& original_box);
    
        //it computes the union of many masks
        void get_mask_union(const std::vector<cv::Mat>& masks, std::vector<int>& idxs, cv::Mat& final_mask);
        
        //it computes the intersection between two masks
        void intersect_masks(const cv::Mat& input_union, const cv::Mat& input_skin, cv::Mat& output);
    
        //it returns the mask associated to the biggest segmented region
        void get_biggest_region(cv::Mat& seg_roi, cv::Mat& mask);
    
        //function used by get_biggest_region
        int count_nozero_pixels(const cv::Mat& mask);



};


bool is_Greyscale(cv::Mat img, int batch_size); // check if the image is grayscale
