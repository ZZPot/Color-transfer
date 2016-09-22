#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string.h>
#include <vector>
#include <math.h>

using namespace cv;

#define WND_NAME_RES "Result Picture"

Mat RGB_to_LMS = (Mat_<double>(3,3) <<  0.3811, 0.5783, 0.0402,
										0.1967, 0.7244, 0.0782,
										0.0241, 0.1288, 0.8444);
Mat LMS_to_RGB = (Mat_<double>(3,3) <<  4.4679, -3.5873, 0.1193,
										-1.2186, 2.3809, -0.1624,
										0.0497, -0.2439, 1.2045);
Mat LMS_to_lab_1 = (Mat_<double>(3,3) << 1/sqrt(3), 0, 0,
										0, 1/sqrt(6), 0,
										0, 0, 1/sqrt(2));
Mat LMS_to_lab_2 = (Mat_<double>(3,3) <<	1, 1, 1,
											1, 1, -2,
											1, -1, 0);
struct ct_image
{
	std::string first;
	std::string second;
	std::string morph;
};
bool makeCT(ct_image images);
Mat convertTolab(Mat input);
Mat convertFromlab(Mat input);

int main()
{
	//transpose(RGB_to_LMS, RGB_to_LMS);
	//transpose(LMS_to_RGB, LMS_to_RGB);
	transpose(LMS_to_lab_1, LMS_to_lab_1);
	transpose(LMS_to_lab_2, LMS_to_lab_2);
	ct_image images_1 = {"pic_1.jpg", "pic_2.jpg", "pic_1_2(1).jpg"};
	if(makeCT(images_1))
	{
		Mat res_pic = imread(images_1.morph);
		imshow(WND_NAME_RES, res_pic);
		waitKey(0);
	}
	return 0;
}

bool makeCT(ct_image images)
{
	Mat img1 = imread(images.first);
	Mat img2 = imread(images.second);

	Mat img1_lab = convertTolab(img1);
	Mat img2_lab = convertTolab(img2);
	

	Mat mean1, mean2, stdd1, stdd2;
	meanStdDev(img1_lab, mean1, stdd1);
	meanStdDev(img2_lab, mean2, stdd2);

	std::vector<Mat> img1_lab_channels, img2_lab_channels;
	split(img1_lab, img1_lab_channels);
	split(img2_lab, img2_lab_channels);

	for(int i = 0; i < 3; i++) 
	{
		img1_lab_channels[i] -= mean1.at<double>(i);
		img1_lab_channels[i] *= (stdd1.at<double>(i) / stdd2.
		at<double>(i));
		img1_lab_channels[i] += mean1.at<double>(i);
	}
	Mat result;
	merge(img1_lab_channels, result);
	result = convertFromlab(result);
	imwrite(images.morph, result);
	return true;
}

Mat convertTolab(Mat input)
{
	Mat img;
	input.convertTo(img,CV_32FC1);
	Mat img_lms;
	transform(img, img_lms, RGB_to_LMS);
	// log10 !!!!!!!!
	Mat img_lab;
	transform(img_lms, img_lab, LMS_to_lab_2);
	transform(img_lab, img_lab, LMS_to_lab_1);
	return img_lab;
}
Mat convertFromlab(Mat input)
{
	Mat img_lms;
	Mat trans1, trans2;
	transpose(LMS_to_lab_1, trans1);
	transpose(LMS_to_lab_2, trans2);
	transform(input, img_lms, trans1);
	transform(img_lms, img_lms, trans2);
	// pow10 !!!!!!!!!
	Mat img_RGB;
	transform(img_lms, img_RGB, LMS_to_RGB);
	img_RGB.convertTo(img_RGB,CV_8UC1);
	return img_RGB;
}