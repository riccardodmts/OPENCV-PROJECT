//evaluation.cpp

#include "evaluation.h"

Evaluation::Evaluation(std::string img_path, std::string true_mask_path, cv::Mat our_mask, std::string bbox_path, std::vector<cv::Rect> our_bbox)
{
    image = cv::imread(img_path);
    true_mask = cv::imread(true_mask_path);
    detected_mask = our_mask;

    std::string filename(bbox_path);
    int number[4];

    std::ifstream input_file(filename);
    if (!input_file.is_open()) {
        std::cerr << "Could not open the file - '"
             << filename << "'" << std::endl;
    }
    //process 4 number at the time
    while (input_file >> number[0] >> number[1] >> number[2] >> number[3]) {
      true_bbox.push_back(cv::Rect(number[0],number[1],number[2],number[3]));
    }
    input_file.close();

    detected_bbox = our_bbox;

    ValidateBoundingBox();

}


Evaluation::Evaluation(std::string img_path, cv::Mat our_mask, std::vector<cv::Rect> our_bbox){
  //in this case I suppose to have the following folder hierarchy
  /*
  test |
       |-> rgb (with the test images)
       |-> mask (with all the true mask)
       |-> det (with all the true bbox)
  */
  std::string temp_path = img_path;
  //we separate all the substring with delimiter "/" and we extract the last substring
  // this will be the name of our image
  int pos = 0;
  std::string token;
  std::string delimiter = "/";
  while ((pos = temp_path.find(delimiter)) != std::string::npos) {
      token = temp_path.substr(0, pos);
      temp_path.erase(0, pos + delimiter.length());
  }

  //now we extract the name without the extension, to determine the mask and bbox name
  std::string img_name_no_ext = temp_path.substr(0, temp_path.find("."));

  std::string truemask_path = "./../../test/mask/" + img_name_no_ext + ".png";
  std::string truebbox_path = "./../../test/det/" + img_name_no_ext + ".txt";

  // we recall the other costructor with all the paths
  new (this) Evaluation(img_path, truemask_path, our_mask, truebbox_path, our_bbox );

}


void Evaluation::ValidateBoundingBox()
{

  std::vector<cv::Rect> temp_true_bbox = true_bbox;
  std::vector<cv::Rect> temp_detected_bbox = detected_bbox;

  int count = 0;
  /*
    for each true bbox we assign our detected bbox that maximize the IoU for that true bbox

    assign, means to order the two vectors (true bbox and detected bbox)
    and consider the i-th true bbox with the correspondent i-th detected bbox
  */
  while(count<detected_bbox.size())
  {
    int true_max = -1, det_max = -1;
    MaximizeIoU(temp_true_bbox, temp_detected_bbox, true_max, det_max);
    true_bbox[count] = temp_true_bbox[true_max];
    detected_bbox[count] = temp_detected_bbox[det_max];
    count++;
    temp_true_bbox.erase(temp_true_bbox.begin() + true_max);
    temp_detected_bbox.erase(temp_detected_bbox.begin() + det_max);
  }

  /*
    if the n. of detected bbox is less than the number of true bbox,
    we add to the detected box vector, the bounding box (from the one we detected)
    that maximize the IoU of the remaining true bbox to be assigned
  */
  while(count<true_bbox.size())
  {
      int true_max = -1, det_max = -1;
      MaximizeIoU(temp_true_bbox, detected_bbox, true_max, det_max);
      true_bbox[count] = temp_true_bbox[true_max];
      detected_bbox.push_back(detected_bbox[det_max]);
      count++;
      temp_true_bbox.erase(temp_true_bbox.begin() + true_max);
  }

}

void Evaluation::MaximizeIoU(std::vector<cv::Rect> true_box, std::vector<cv::Rect> det_bbox, int &true_max, int &det_max)
{
  std::vector<std::vector<float>> iou_bbox;
  for(int i=0;i<true_box.size();i++)
  {
    std::vector<float> temp;
    for(int j=0;j<det_bbox.size();j++)
      temp.push_back(IoU(true_box[i],det_bbox[j]));

    iou_bbox.push_back(temp);
  }

  float iou_max = -1;
  true_max= -1;
  det_max = -1;
  for(int i=0;i<iou_bbox.size();i++)
    for(int j=0;j<iou_bbox[i].size();j++)
      if(iou_bbox[i][j] > iou_max)
      {
        iou_max = iou_bbox[i][j];
        true_max = i;
        det_max = j;
      }
}




float Evaluation::IoU(cv::Rect bbox_A, cv::Rect bbox_B)
{
  int xA = std::max(bbox_A.x, bbox_B.x);
	int yA = std::max(bbox_A.y, bbox_B.y);
	int xB = std::min(bbox_A.x + bbox_A.width, bbox_B.x + bbox_B.width);
	int yB = std::min(bbox_A.y + bbox_A.height, bbox_B.y + bbox_B.height);

	//compute the area of intersection rectangle
	int interArea = std::max(0, xB - xA + 1) * std::max(0, yB - yA + 1);

	//compute the area of both the prediction and ground-truth rectangles
	int boxAArea = (bbox_A.width+1) * (bbox_A.height+1);
	int boxBArea = (bbox_B.width+1) * (bbox_B.height+1);

	//compute the intersection over union by taking the intersection
	//area and dividing it by the sum of prediction + ground-truth areas - the interesection area
	float iou = interArea / float(boxAArea + boxBArea - interArea);
	//return the intersection over union value
	return iou;
}

cv::Mat Evaluation::IoU()
{
  cv::Mat iou_img = image.clone();
  float iou;
  for(int i=0;i<true_bbox.size();i++)
  {
    //drawing the bounding boxes
    cv::rectangle(iou_img, true_bbox.at(i),cv::Scalar(255,0,0),2);
    cv::rectangle(iou_img, detected_bbox.at(i),cv::Scalar(0,0,255),2);

    iou = IoU(true_bbox[i],detected_bbox[i]);

    //adding the text of the IoU on the image
    cv::putText(iou_img, //target image
            "IoU = " + std::to_string(iou), //text
            cv::Point(true_bbox[i].x-10, true_bbox[i].y-5), //top-left position
            cv::FONT_HERSHEY_DUPLEX,
            0.5, //font scale
            CV_RGB(255,255,255), //font color
            2);

  }
  return iou_img;
}



void Evaluation::PixelAccuracy(){

  //size check
  if(detected_mask.rows != true_mask.rows || detected_mask.cols != true_mask.cols)
    return;

  cv::Mat accuracy_img = image.clone();
  cv::Mat gray_true_mask, gray_det_mask;

  // we convert the mask images into gray scale to simplify comparison
  cv::cvtColor(true_mask, gray_true_mask, cv::COLOR_BGR2GRAY);
  cv::cvtColor(detected_mask, gray_det_mask, cv::COLOR_BGR2GRAY);

  int false_positive=0, false_negative = 0;
  int true_positive=0, true_negative = 0;
  int positive = 0, negative = 0;
  for(int i=0; i < true_mask.rows; i++)
    for(int j=0; j < true_mask.cols; j++)
    {

      int color_ch_modified = int( gray_det_mask.at<uchar>(i,j) ) % 3;

      if(int( gray_true_mask.at<uchar>(i,j) ) == 0  && int( gray_det_mask.at<uchar>(i,j) )== 0)
      {
        true_negative++;
        negative++;
      }

      else if(int( gray_true_mask.at<uchar>(i,j) ) != 0  && int( gray_det_mask.at<uchar>(i,j) )!= 0)
      {
        true_positive++;
        positive++;
        accuracy_img.at<cv::Vec3b>(i,j) = detected_mask.at<cv::Vec3b>(i,j);
        //accuracy_img.at<cv::Vec3b>(i,j)[color_ch_modified]+= int( gray_det_mask.at<uchar>(i,j) ) /3;
      }

      else if(int( gray_true_mask.at<uchar>(i,j) ) != 0  && int( gray_det_mask.at<uchar>(i,j) ) == 0)
      {
        false_negative++;
        positive++;
      }

      else if(int( gray_true_mask.at<uchar>(i,j) ) == 0  && int( gray_det_mask.at<uchar>(i,j) ) != 0)
      {
        false_positive++;
        negative++;
        accuracy_img.at<cv::Vec3b>(i,j) = detected_mask.at<cv::Vec3b>(i,j);
        //accuracy_img.at<cv::Vec3b>(i,j)[color_ch_modified]+= int( gray_det_mask.at<uchar>(i,j) ) /3;
      }

    }
    int total_pop = detected_mask.rows*detected_mask.cols;
    std::cout<<total_pop<<std::endl;
    std::cout<<"False Positive: "<<false_positive<<std::endl;
    std::cout<<"False Negative: "<<false_negative<<std::endl;
    std::cout<<"True Positive: "<<true_positive<<std::endl;
    std::cout<<"True Negative: "<<true_negative<<std::endl;

    float accuracy = (true_positive + true_negative)/float(total_pop);
    float precision = float(true_positive)/(true_positive + false_positive);
    float recall = float(true_positive)/(true_positive + false_negative);

    std::cout<<"accuracy: "<<accuracy<<" precision: "<<precision<<" recall: "<<recall<<std::endl;

    float tp_rate = float(true_positive)/positive;
    float tn_rate = float(true_negative)/negative;
    float balance_accuracy = (tp_rate + tn_rate)/2;

    std::cout<<"balance accuracy: "<<balance_accuracy<<std::endl;



    float F1 = 2*(precision*recall)/(precision+recall);
    std::cout<<"F1 = "<<F1<<std::endl;

    cv::imshow("A",accuracy_img);
    cv::waitKey(0);
}
