//segmentation.cpp

#include "skin_segmentation.h"

cv::Mat get_skin(cv::Mat Input_image)
{
    cv::Mat RGB = Input_image;
    cv::Mat HSV;
    cv::cvtColor(RGB, HSV, cv::COLOR_BGR2HSV);
    cv::Mat YCBCR;
    cv::cvtColor(RGB, YCBCR, cv::COLOR_BGR2YCrCb);

    cv::Mat out = Input_image;

    //(H : Hue ; S: Saturation ; R : Red ; B: Blue ; G : Green ; Cr, Cb : Chrominance components ; Y : luminance component )
    float R, G, B, H, S, V, Y, CB, CR;

    for (int y = 0; y < Input_image.rows; y++)
    {
		for (int x = 0; x < Input_image.cols; x++)
        {

            B = RGB.at<cv::Vec3b>(y, x)[0];
            G = RGB.at<cv::Vec3b>(y, x)[1];
            R = RGB.at<cv::Vec3b>(y, x)[2];

            H = HSV.at<cv::Vec3b>(y, x)[0];
            S = HSV.at<cv::Vec3b>(y, x)[1];
            V = HSV.at<cv::Vec3b>(y, x)[2];

            Y = YCBCR.at<cv::Vec3b>(y, x)[0];
            CR = YCBCR.at<cv::Vec3b>(y, x)[1];
            CB = YCBCR.at<cv::Vec3b>(y, x)[2];

            //https://arxiv.org/ftp/arxiv/papers/1708/1708.02694.pdf relative paper 

            if((0.0 <= H <= 50.0 && 30 <= S <= 150 && R > 95 && G > 40 && B > 20 && R > G && R > B && abs(R - G) > 15) ||
                (R > 95 && G > 40 && B > 20 && R > G && R > B && abs(R - G) > 15 && CR > 135 && CB > 85 && 
                Y > 80 && CR <= (1.5862*CB)+20 && CR>=(0.3448*CB)+76.2069 && CR >= (-4.5652*CB)+234.5652 && 
                CR <= (-1.15*CB)+301.75 && CR <= (-2.2857*CB)+432.85))
            {
                out.at<cv::Vec3b>(y, x)[0] = B;
                out.at<cv::Vec3b>(y, x)[1] = G;
                out.at<cv::Vec3b>(y, x)[2] = R;
            }

            else
            {
                out.at<cv::Vec3b>(y, x)[0] = 0;
                out.at<cv::Vec3b>(y, x)[1] = 0;
                out.at<cv::Vec3b>(y, x)[2] = 0;
            }
        }
    }

    return out;
}




cv::Mat K_Means(cv::Mat Input, int K, cv::Mat &RGB_centers) {

	cv::Mat samples(Input.rows * Input.cols, Input.channels(), CV_32F);
	for (int y = 0; y < Input.rows; y++)
		for (int x = 0; x < Input.cols; x++)
			for (int z = 0; z < Input.channels(); z++)
				if (Input.channels() == 3) {
					samples.at<float>(y + x * Input.rows, z) = Input.at<cv::Vec3b>(y, x)[z];
				}
				else {
					samples.at<float>(y + x * Input.rows, z) = Input.at<uchar>(y, x);
				}


	cv::Mat labels;
	int attempts = 5;
	cv::Mat centers;
	cv::kmeans( samples, K, labels, cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 10000, 0.0001),
            attempts, cv::KMEANS_PP_CENTERS, centers );

    RGB_centers = centers;

	cv::Mat new_image(Input.size(), Input.type());
	for (int y = 0; y < Input.rows; y++)
		for (int x = 0; x < Input.cols; x++)
		{
			int cluster_idx = labels.at<int>(y + x * Input.rows, 0);
			if (Input.channels()==3) {
				for (int i = 0; i < Input.channels(); i++) {
					new_image.at<cv::Vec3b>(y, x)[i] = centers.at<float>(cluster_idx, i);
				}
			}
			else {
				new_image.at<uchar>(y, x) = centers.at<float>(cluster_idx, 0);
			}
		}
        
	return new_image;
}