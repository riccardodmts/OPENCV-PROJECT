//get_segmentation.h

#ifndef _DET_
#define _DET_
#include "detection.h"
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
    

    //PRIVATE METHODS

    private:

        void get_masks_per_region(cv::Mat& seg_roi, std::vector<cv::Mat>& masks);
        void get_expanded_roi(const cv::Mat& img, const cv::Rect& roi, cv::Rect& new_roi, int max_disp);
        void get_mask_original_size(const cv::Rect& roi, const cv::Rect& new_roi, const cv::Mat& mask, cv::Mat& new_mask);


};