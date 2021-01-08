#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string.h>
#include <vector>
#include <math.h>
#include <conio.h>


#define WND_NAME_RES "Result Picture"
#define WND_NAME_SOURCE "Source Picture"

/* For escaping zeros*/
#define EPSILON			0.07
#define RED_MUL			0.21
#define GREEN_MUL		0.72
#define BLUE_MUL		0.07

cv::Mat RGB_to_LMS = (cv::Mat_<float>(3,3) <<	0.3811f, 0.5783f, 0.0402f,
										0.1967f, 0.7244f, 0.0782f,
										0.0241f, 0.1288f, 0.8444f);

cv::Mat LMS_to_RGB = (cv::Mat_<float>(3,3) <<	4.4679f, -3.5873f, 0.1193f,
										-1.2186f, 2.3809f, -0.1624f,
										0.0497f, -0.2439f, 1.2045f);

cv::Mat LMS_to_lab_1 = (cv::Mat_<float>(3,3) << 1/sqrt(3), 0, 0,
										0, 1/sqrt(6), 0,
										0, 0, 1/sqrt(2));
cv::Mat LMS_to_lab_2 = (cv::Mat_<float>(3,3) << 1, 1, 1,
										1, 1, -2,
										1, -1, 0);

float _x = 1/sqrt(3), _y = 1/sqrt(6), _z = 1/sqrt(2);
cv::Mat LMS_to_lab = (cv::Mat_<float>(3,3) <<	_x, _x, _x,
										_y, _y, -2*_y,
										_z, -_z, 0);
cv::Mat lab_to_LMS = (cv::Mat_<float>(3,3) <<	_x, _y, _z,
										_x, _y, -_z,
										_x, -2*_y, 0);

struct ct_image
{
	std::string source;
	std::string target;
	std::string result;
};

ct_image images[] = {
	{"images/1/img_1.jpg", "images/1/img_2.jpg", "images/1/img_1_2_cv.jpg"},
	{"images/2/img_3.jpg", "images/2/img_4.jpg", "images/2/img_3_4_cv.jpg"},
	{"images/3/pic_1.jpg", "images/3/pic_2.jpg", "images/3/pic_1_2_cv.jpg"},
	{"images/4/1.jpg", "images/4/2.jpg", "images/4/1_2_cv.jpg"}};

bool makeCT(ct_image images);
bool makeCTCIE(ct_image images);

cv::Mat convertTolab(cv::Mat input);
cv::Mat convertFromlab(cv::Mat input);
cv::Mat _transform(cv::Mat mat, cv::Mat core);

void showMinStd(cv::Mat input, std::string caption);

//#define SINGLE_MATRIX
#define PER_CHANNEL
void showMat(cv::Mat mat)
{
	for(int i = 0; i < mat.rows; i++)
	{
		for(int j = 0; j < mat.cols; j++)
		{
			printf("%02.5f\t", mat.at<float>(i, j));
		}
		printf("\n");
	}
}
int main()
{
	unsigned img_pack = 3;

	if(makeCT(images[img_pack]))
	{
		cv::Mat res_pic = cv::imread(images[img_pack].result);
		cv::imshow(WND_NAME_RES, res_pic);
		cv::waitKey(0);
	}
	return 0;
}
bool makeCT(ct_image images)
{
	cv::Mat imgs = cv::imread(images.source);
	cv::Mat imgt = cv::imread(images.target);
	cv::Mat imgs_lab;
	cv::Mat imgt_lab;
	imgs_lab = convertTolab(imgs);
	imgt_lab = convertTolab(imgt);

	cv::Mat meant, means, stddt, stdds;
	meanStdDev(imgs_lab, means, stdds);
	meanStdDev(imgt_lab, meant, stddt);
	cv::Mat result;
#ifdef PER_CHANNEL
	std::vector<cv::Mat> imgs_lab_channels, imgt_lab_channels;
	split(imgs_lab, imgs_lab_channels);
	split(imgt_lab, imgt_lab_channels);
	for(int i = 0; i < 3; i++)
	{
		double koef;
		if(stdds.at<double>(i) == 0)
		{
			continue; // leave it same.
		}
		if(stddt.at<double>(i) == stdds.at<double>(i))
			koef = 1;
		else
			koef = stddt.at<double>(i)/stdds.at<double>(i);
		imgs_lab_channels[i] = (imgs_lab_channels[i] - means.at<double>(i)) *
			koef + meant.at<double>(i);
	}
	merge(imgs_lab_channels, result);
#else
	Scalar	mean_src(means.at<double>(0), means.at<double>(1), means.at<double>(2)),
			mean_tg(meant.at<double>(0), meant.at<double>(1), meant.at<double>(2)),
			koef(	stddt.at<double>(0)/stdds.at<double>(0),
					stddt.at<double>(1)/stdds.at<double>(1),
					stddt.at<double>(2)/stdds.at<double>(2));
	multiply((imgs_lab - mean_src), koef, imgs_lab);
	result = imgs_lab + mean_tg;
#endif
	result = convertFromlab(result);
	imwrite(images.result, result);
	return true;
}
bool makeCTCIE(ct_image images)
{
	cv::Mat imgs = cv::imread(images.source);
	cv::Mat imgt = cv::imread(images.target);
	cv::Mat imgs_lab;
	cv::Mat imgt_lab;
	cv::cvtColor(imgs, imgs_lab, cv::COLOR_BGR2Lab);
	cv::cvtColor(imgt, imgt_lab, cv::COLOR_BGR2Lab);
	imgs.convertTo(imgs, CV_32FC1);
	imgt.convertTo(imgt, CV_32FC1);

	cv::Mat meant, means, stddt, stdds;
	meanStdDev(imgs_lab, means, stdds);
	meanStdDev(imgt_lab, meant, stddt);

	std::vector<cv::Mat> imgs_lab_channels, imgt_lab_channels;
	split(imgs_lab, imgs_lab_channels);
	split(imgt_lab, imgt_lab_channels);

	for(int i = 0; i < 3; i++)
	{
		imgs_lab_channels[i] = (imgs_lab_channels[i] - means.at<double>(i)) *
		stddt.at<double>(i)/stdds.at<double>(i) + meant.at<double>(i);
	}

	cv::Mat result;
	merge(imgs_lab_channels, result);
	result.convertTo(result, CV_8UC1);
	cv::cvtColor(result, result, cv::COLOR_Lab2BGR);
	cv::imwrite(images.result, result);
	return true;
}
cv::Mat convertTolab(cv::Mat input)
{
	cv::Mat img_RGB;
	cv::cvtColor(input, img_RGB, cv::COLOR_BGR2RGB);
	cv::Scalar min_scalar(EPSILON, EPSILON, EPSILON);
	cv::Mat min_mat = cv::Mat(img_RGB.size(), CV_32FC3, min_scalar);
	img_RGB.convertTo(img_RGB, CV_32FC1, 1/255.f);
	// To escape zeros in LMS color space, we need to replace every black pixel
	//max(img_RGB, min_mat, img_RGB); // before converting to LMS
	showMinStd(img_RGB, "initial");
	cv::Mat img_lms;
	transform(img_RGB, img_lms, RGB_to_LMS);
	max(img_lms, min_mat, img_lms); // right before log10
	showMinStd(img_lms, "after convert to LMS");
	// this trick from Jun Yan's code
	// log10(x)=ln(x)/ln(10)
	cv::log(img_lms,img_lms);
	img_lms /= log(10);
	showMinStd(img_lms, "after log10");
	cv::Mat img_lab;
#ifdef SINGLE_MATRIX ˜CV_LOAD_IMAGE_GRAYSCALE
	transform(img_lms, img_lab, LMS_to_lab);
#else
	transform(img_lms, img_lab, LMS_to_lab_1 * LMS_to_lab_2);
#endif
	return img_lab;
}
cv::Mat convertFromlab(cv::Mat input)
{
	cv::Mat img_lms;
#ifdef SINGLE_MATRIX
	transform(input, img_lms, lab_to_LMS);
#else
	transpose(LMS_to_lab_2, LMS_to_lab_2);
	transform(input, img_lms, LMS_to_lab_2 * LMS_to_lab_1);
#endif
	// this trick from Jun Yan's code
	// 10^x=(e^x)^(ln10)
	exp(img_lms,img_lms);
	pow(img_lms,log(10),img_lms);

	cv::Mat img_RGB;
	transform(img_lms, img_RGB, LMS_to_RGB);
	img_RGB.convertTo(img_RGB, CV_8UC1, 255.f);
	cv::Mat img_BGR;
	cv::cvtColor(img_RGB, img_BGR, cv::COLOR_RGB2BGR);
	return img_BGR;
}
void showMinStd(cv::Mat input, std::string caption)
{
	cv::Mat mean, stdd;
	cv::meanStdDev(input, mean, stdd);
	printf("%s\n", caption.c_str());
	for(int i = 0; i < input.channels(); i++)
	{
		printf("\t%d %f %f\n", i, mean.at<double>(i), stdd.at<double>(i));
	}
}