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

    input_path = "./../../pr1.ppm";
    output_path = "./../../segmentated/prova.ppm";

}


//with hand detector
HandSegmentor::HandSegmentor(HandDetector* ptrHandDetector) : ptrHandDetector(ptrHandDetector){

    sigma = 0.5;
    k = 1000;
    min_size = 100;

    input_path = "./../../pr1.ppm";
    output_path = "./../../segmentated/prova.ppm";

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

int HandSegmentor::test_region(const std::vector<cv::Rect>& boxes, const cv::Rect& original_box){

    int original_area = original_box.width * original_box.height;
    float sum_area = 0;

    for(size_t i = 0; i < boxes.size(); i++)
    {
        if( (float)(boxes[i].width * boxes[i].height) > (float)(0.5 * original_area) )
            return 1;

        sum_area = sum_area + (float)(boxes[i].width * boxes[i].height);
    }

    if(sum_area > (float)(0.6 * original_area))
        return 1;
    else
        return 0;

}

void HandSegmentor::get_idx_of_regions(const cv::Mat& roi_img, const std::vector<cv::Mat>& masks, std::vector<int>& idxs, const cv::Rect& original_box){

    idxs.reserve(masks.size());

    for(size_t i = 0; i < masks.size(); i++)
    {
        cv::Mat temp;
        roi_img.copyTo(temp, masks[i]);
        std::vector<cv::Rect> boxes;
        std::vector<float> conf_values;
        ptrHandDetector->detect_hands(temp, conf_values, boxes);

        if(test_region(boxes, original_box))
            idxs.push_back(i);
    }
}

void HandSegmentor::get_mask_original_size(const cv::Rect& roi, const cv::Rect& new_roi, const std::vector<cv::Mat>& masks, std::vector<cv::Mat>& new_masks){

    for(size_t i = 0; i < masks.size(); i++)
    {
        cv::Mat temp;
        get_mask_original_size(roi, new_roi, masks[i], temp);
        new_masks.push_back(temp);
    }

}

void HandSegmentor::get_mask_union(const std::vector<cv::Mat>& masks, std::vector<int>& idxs, cv::Mat& final_mask){

    int row = masks[0].rows;
    int col = masks[0].cols;
    final_mask = cv::Mat::zeros(row, col, CV_8UC1);

    for(size_t i = 0; i < idxs.size(); i++)
    {
        for(size_t r = 0; r < row; r++)
        {
            for(size_t c = 0; c < col; c++)
            {
                if(masks[idxs[i]].at<uchar>(r,c) == 255)
                    final_mask.at<uchar>(r,c) = 255;
            }
        }
    }
}

void HandSegmentor::final_masks(const char* path, std::vector<cv::Rect>& boxes, std::vector<cv::Mat>& masks){

    cv::Mat img = cv::imread(path);
    cv::imwrite("./../../pr1.ppm", img);// check path

    get_segmentation();

    cv::Mat segmented = cv::imread("./../../segmentated/prova.ppm");

    bool is_gray = is_Greyscale(img, 30);
    int max_disp = 20;

    for(size_t i = 0; i < boxes.size(); i++)
    {
        cv::Rect new_box;
        get_expanded_roi(img, boxes[i], new_box, max_disp);
        cv::Mat img_cropped = img(new_box);
        cv::Mat seg_cropped = segmented(new_box);
        std::vector<cv::Mat> masks_per_region;
        get_masks_per_region(seg_cropped, masks_per_region);
        std::vector<int> idxs;
        get_idx_of_regions(img_cropped, masks_per_region, idxs, boxes[i]);

        cv::Mat final_mask; //final result for current bbox

        if(idxs.size() == 0 && !is_gray)
        {
            cv::Mat skin = get_skin(img(boxes[i]));

            from_skin_to_mask(skin, final_mask);
            masks.push_back(final_mask);
        }

        else if(idxs.size() > 0)
        {
            std::vector<cv::Mat> resized;
            get_mask_original_size(boxes[i], new_box, masks_per_region, resized);
            cv::Mat mask_union;
            get_mask_union(resized, idxs, mask_union);

            if(!is_gray)
            {
                cv::Mat skin = get_skin(img(boxes[i]));
                cv::Mat mask_from_skin;
                from_skin_to_mask(skin, mask_from_skin);
                intersect_masks(mask_union, mask_from_skin, final_mask);
                masks.push_back(final_mask);
            }

            else
            masks.push_back(mask_union);
        }

        else
        {
            cv::Mat bad_mask = cv::Mat(boxes[i].height, boxes[i].width, CV_8UC1, cv::Scalar(255));
            masks.push_back(bad_mask);
        }
    }
}

bool is_Greyscale(cv::Mat img, int batch_size){

  for(int i=0;i<batch_size;i++)
  {
    int rand_rows = rand() % img.rows + 1;
    int rand_cols = rand() % img.cols + 1;

    if( int(img.at<cv::Vec3b>(rand_rows,rand_cols)[0]) != int(img.at<cv::Vec3b>(rand_rows,rand_cols)[1]) || int(img.at<cv::Vec3b>(rand_rows,rand_cols)[1]) != int(img.at<cv::Vec3b>(rand_rows,rand_cols)[2]))
      return false;

  }
  return true;
}

void HandSegmentor::intersect_masks(const cv::Mat& input_union, const cv::Mat& input_skin, cv::Mat& output){

    int row = input_union.rows;
    int col = input_union.cols;
    output = cv::Mat::zeros(row, col, CV_8UC1);


    for(size_t r = 0; r < row; r++)
    {
        for(size_t c = 0; c < col; c++)
        {
            if((input_union.at<uchar>(r,c) == 255) && (input_skin.at<uchar>(r,c) == 255))
                output.at<uchar>(r,c) = 255;
        }
    }

}

void HandSegmentor::from_skin_to_mask(const cv::Mat& skin_output, cv::Mat& output){

    int rows = skin_output.rows;
    int cols = skin_output.cols;

    output = cv::Mat::zeros(rows, cols, CV_8UC1);
    for(int i = 0; i < rows; i++){

        for(int j = 0; j < cols; j++){

           if((skin_output.at<cv::Vec3b>(i, j)[0] == 0.0f)&&(skin_output.at<cv::Vec3b>(i, j)[1] == 0.0f)&&(skin_output.at<cv::Vec3b>(i, j)[2] == 0.0f));
           else output.at<uchar>(i,j) = 255;

        }
    }


}

cv::Mat HandSegmentor::final_mask(const char* path, std::vector<cv::Rect> boxes)
{
  std::vector<cv::Mat> masks;

  final_masks(path, boxes, masks);

  cv::Mat img = cv::imread(path);
  int row = img.rows;
  int col = img.cols;
  cv::Mat output = cv::Mat::zeros(row, col, CV_8UC3);

  for(int i=0;i<boxes.size();i++)
  {
    cv::Mat temp;
    if(masks[i].channels() != 1)
      cv::cvtColor(masks[i], temp, cv::COLOR_BGR2GRAY);
    else temp = masks[i];

    int B = rand() % 255;
    int R = rand() % 255;
    int G = rand() % 255;
    for(int r = 0; r < temp.rows; r++)
      for(int c = 0; c < temp.cols; c++)
      {
        if(int(temp.at<uchar>(r,c)) != 0)
        {
          output.at<cv::Vec3b>(boxes[i].y + r,boxes[i].x + c)[0] = B;
          output.at<cv::Vec3b>(boxes[i].y + r,boxes[i].x + c)[1] = R;
          output.at<cv::Vec3b>(boxes[i].y + r,boxes[i].x + c)[2] = G;
        }
      }
  }

  return output;
}

void HandSegmentor::get_biggest_region(cv::Mat& seg_roi, cv::Mat& mask){

    std::vector<cv::Mat> masks;

    get_masks_per_region(seg_roi, masks);

    int n_nozero_pixels = 0;
    int idx = 0;
    int temp;
    for(size_t i = 1; i < masks.size(); i++){
        temp = count_nozero_pixels(masks[i]);

        if( temp > n_nozero_pixels){

            idx = i;
            n_nozero_pixels = temp;

        }
    }

    mask = masks[idx].clone();


}

int HandSegmentor::count_nozero_pixels(const cv::Mat& mask){

    int counter = 0;

    for(size_t i = 0; i < mask.rows; i++){
        for(size_t j = 0; j < mask.cols; j++){

            if((int)(mask.at<uchar>(i,j)) > 0)  counter ++;           
        }
    }
    
    return counter;

}
