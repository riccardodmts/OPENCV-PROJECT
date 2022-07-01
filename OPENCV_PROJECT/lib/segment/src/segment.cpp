//segment.cpp

#ifndef _SEGMENT_
#define _SEGMENT_

#include <cstdio>
#include <cstdlib>
#include "image.h"
#include "misc.h"
#include "pnmfile.h"
#include "segment-image.h"

#include "get_segmentation.h"

#endif

//CONSTRUCTORS


//default one
HandSegmentor::HandSegmentor(){

    sigma = 0.5;
    k = 800;
    min_size = 30;

    input_path = "./../../../pr1.ppm";
    output_path = "./../../../segmentated/prova.ppm";

}


//with hand detector
HandSegmentor::HandSegmentor(HandDetector* ptrHandDetector) : ptrHandDetector(ptrHandDetector){

    sigma = 0.5;
    k = 1000;
    min_size = 100;

    input_path = "./../../../pr1.ppm";
    output_path = "./../../../segmentated/prova.ppm";

}


//with hand detector and tunable parameters
HandSegmentor::HandSegmentor(HandDetector* ptrHandDetector, float sigma, float k, int min_size, 
                             const char *input_path, const char *output_path) : ptrHandDetector(ptrHandDetector), sigma(sigma), k(k), 
                             min_size(min_size), input_path(input_path), output_path(output_path){

}


//PUBLIC MEMBERS

void HandSegmentor::get_segmentation()
{
        
    printf("loading input image.\n");
    image<rgb> *input = loadPPM(input_path);
        
    printf("processing\n");

    int num_ccs; 
    image<rgb> *seg = segment_image(input, sigma, k, min_size, &num_ccs); 
    savePPM(seg, output_path);

}

void HandSegmentor::test(){

    cv::Mat segmented = cv::imread(output_path);
    cv::Mat img = cv::imread(input_path);

    //cv::Mat output;
    //ptrHandDetector->detect_hands(img, output);

    //cv::imshow("test", output);


    std::vector<cv::Rect> boxes;
    std::vector<float> conf;
    ptrHandDetector->detect_hands(img, conf, boxes);
    int idx = 0;
    int max_disp = 20;
    cv::Rect box;
    get_expanded_roi(img, boxes[idx], box, max_disp);
    std::cout<<std::endl<<boxes[idx].x<<  " " << boxes[idx].y<< " " << boxes[idx].width << " " << boxes[idx].height<<std::endl; 
    std::cout<<std::endl<<box.x<<  " " << box.y<< " " << box.width << " " << box.height<<std::endl; 
    
    std::vector<cv::Mat> masks;

    cv::Mat roi = segmented(box);
    cv::Mat roi_original = img(box);
    get_masks_per_region(roi, masks);

    cv::imshow("roi before cut", roi_original);
    std::cout<<roi_original.size()<<std::endl;
    cv::Mat output_cut;
    get_mask_original_size(boxes[idx], box, roi_original, output_cut);
    cv::imshow("after cutt", output_cut);
    std::cout<<output_cut.size()<<std::endl;
    cv::waitKey(0);
    for(int i = 0; i < masks.size(); i++){

        cv::Mat temp;
        roi_original.copyTo(temp, masks[i]);
        cv::Mat out;
        ptrHandDetector->detect_hands(temp, out);
        cv::imshow("res", out);
        cv::waitKey(0);

    }


}

//PRIVATE METHODS
void HandSegmentor::get_masks_per_region(cv::Mat& seg_roi, std::vector<cv::Mat>& masks){

    int rows = seg_roi.rows;
    int cols = seg_roi.cols;
    double B;
    double G;
    double R;
    int test = 0;
    int index;

    std::vector<cv::Scalar> regions_color;

    for(int i = 0; i < rows; i++){

        for(int j = 0; j < cols; j++){

            
            if((i == 0) && (j == 0)){
                regions_color.push_back(cv::Scalar(seg_roi.at<cv::Vec3b>(i,j)[0],seg_roi.at<cv::Vec3b>(i,j)[1], seg_roi.at<cv::Vec3b>(i,j)[2]));
                masks.push_back(cv::Mat::zeros(rows, cols, CV_8UC1));
                masks[0].at<uchar>(0,0) = (uchar)(255);

            
            }
            else{

                B = seg_roi.at<cv::Vec3b>(i,j)[0];
                G = seg_roi.at<cv::Vec3b>(i,j)[1];
                R = seg_roi.at<cv::Vec3b>(i,j)[2];
                
                for(int k = 0; k < masks.size(); k++){
  

                    if((B == regions_color[k][0])&&(G == regions_color[k][1])&&(R == regions_color[k][2])){
                        index = k;
                        test  = 1;
                        break;
                    }            
                }

                if(test){
                    test = 0;
                    masks[index].at<uchar>(i,j) = (uchar)(255);
                }
                else{

                    regions_color.push_back(cv::Scalar(B,G,R) );
                    masks.push_back(cv::Mat::zeros(rows, cols, CV_8UC1));
                    masks[0].at<uchar>(i,j) = (uchar)(255);
                }
            }
            
        }
        
    }

}


void HandSegmentor::get_expanded_roi(const cv::Mat& img, const cv::Rect& roi, cv::Rect& new_roi, int max_dips){

    int w = roi.width;
    int h = roi.height;
    int x = roi.x;
    int y = roi.y;

    int diff_x = img.cols - x - w;
    int diff_y = img.rows - y - h;

    std::vector<int> to_order;
    to_order.reserve(4);
    to_order.push_back(x);
    to_order.push_back(y);
    to_order.push_back(diff_y);
    to_order.push_back(diff_x);

    sort(to_order.begin(), to_order.end());
    if(to_order[0] < max_dips) max_dips = to_order[0];

    cv::Size disp(max_dips*2, max_dips*2);
    cv::Point translation(-max_dips, -max_dips);

    new_roi = roi;
    new_roi += disp;
    new_roi += translation;


}

void HandSegmentor::get_mask_original_size(const cv::Rect& roi, const cv::Rect& new_roi, const cv::Mat& mask, cv::Mat& new_mask){

    int disp = roi.x - new_roi.x;
    cv::Rect cut(disp, disp, roi.width, roi.height);
    new_mask = mask(cut);

}











