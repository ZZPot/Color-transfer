#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <string.h>
#include <vector>
#include <math.h>
#include <conio.h>

using namespace cv;

#define WND_NAME_RES "Result Picture"
#define WND_NAME_SOURCE "Source Picture"

/* For escaping zeros*/
#define EPSILON			0.07
#define RED_MUL			0.21
#define GREEN_MUL		0.72
#define BLUE_MUL		0.07

Mat RGB_to_LMS = (Mat_<float>(3,3) <<	0.3811f, 0.5783f, 0.0402f,
										0.1967f, 0.7244f, 0.0782f,
										0.0241f, 0.1288f, 0.8444f);

Mat LMS_to_RGB = (Mat_<float>(3,3) <<	4.4679f, -3.5873f, 0.1193f,
										-1.2186f, 2.3809f, -0.1624f,
										0.0497f, -0.2439f, 1.2045f);

Mat LMS_to_lab_1 = (Mat_<float>(3,3) << 1/sqrt(3), 0, 0,
										0, 1/sqrt(6), 0,
										0, 0, 1/sqrt(2));
Mat LMS_to_lab_2 = (Mat_<float>(3,3) << 1, 1, 1,
										1, 1, -2,
										1, -1, 0);

float _x = 1/sqrt(3), _y = 1/sqrt(6), _z = 1/sqrt(2);
Mat LMS_to_lab = (Mat_<float>(3,3) <<	_x, _x, _x,
										_y, _y, -2*_y,
										_z, -_z, 0);
Mat lab_to_LMS = (Mat_<float>(3,3) <<	_x, _y, _z,
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
	{"images/2/img_3.jpg", "images/2/img_4.jpg", "images/2/img_3_4_cv1.jpg"},
	{"images/3/pic_1.jpg", "images/3/pic_2.jpg", "images/3/pic_1_2_cv.jpg"}};

bool makeCT(ct_image images);
bool makeCTCIE(ct_image images);

Mat convertTolab(Mat input);
Mat convertFromlab(Mat input);
Mat _transform(Mat mat, Mat core);

void showMinStd(Mat input, std::string caption);

//#define SINGLE_MATRIX
#define PER_CHANNEL
void showMat(Mat mat)
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
	unsigned img_pack = 1;

	if(makeCT(images[img_pack]))
	{
		Mat res_pic = imread(images[img_pack].result);
		imshow(WND_NAME_RES, res_pic);
		waitKey(0);
	}
	return 0;
}
bool makeCT(ct_image images)
{
	Mat imgs = imread(images.source);
	Mat imgt = imread(images.target);
	Mat imgs_lab;
	Mat imgt_lab;
	imgs_lab = convertTolab(imgs);
	imgt_lab = convertTolab(imgt);

	Mat meant, means, stddt, stdds;
	meanStdDev(imgs_lab, means, stdds);
	meanStdDev(imgt_lab, meant, stddt);
	Mat result;
#ifdef PER_CHANNEL
	std::vector<Mat> imgs_lab_channels, imgt_lab_channels;
	split(imgs_lab, imgs_lab_channels);
	split(imgt_lab, imgt_lab_channels);
	for(int i = 0; i < 3; i++)
	{
		double koef;
		if(stddt.at<double>(i) == stdds.at<double>(i))
			koef = 1;
		else
			koef = stddt.at<double>(i)/stdds.at<double>(i);
		if(stdds.at<double>(i) == 0)
		{
			continue; // leave it same.
		}
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
	Mat imgs = imread(images.source);
	Mat imgt = imread(images.target);
	Mat imgs_lab;
	Mat imgt_lab;
	cvtColor(imgs, imgs_lab, CV_BGR2Lab);
	cvtColor(imgt, imgt_lab, CV_BGR2Lab);
	imgs.convertTo(imgs, CV_32FC1);
	imgt.convertTo(imgt, CV_32FC1);

	Mat meant, means, stddt, stdds;
	meanStdDev(imgs_lab, means, stdds);
	meanStdDev(imgt_lab, meant, stddt);

	std::vector<Mat> imgs_lab_channels, imgt_lab_channels;
	split(imgs_lab, imgs_lab_channels);
	split(imgt_lab, imgt_lab_channels);

	for(int i = 0; i < 3; i++)
	{
		imgs_lab_channels[i] = (imgs_lab_channels[i] - means.at<double>(i)) *
		stddt.at<double>(i)/stdds.at<double>(i) + meant.at<double>(i);
	}

	Mat result;
	merge(imgs_lab_channels, result);
	result.convertTo(result, CV_8UC1);
	cvtColor(result, result, CV_Lab2BGR);
	imwrite(images.result, result);
	return true;
}
Mat convertTolab(Mat input)
{
	Mat img_RGB;
	cvtColor(input, img_RGB, CV_BGR2RGB);
	Scalar min_scalar(EPSILON, EPSILON, EPSILON);
	Mat min_mat = Mat::Mat(img_RGB.size(), CV_32FC3, min_scalar);
	img_RGB.convertTo(img_RGB, CV_32FC1, 1/255.f);
	// To escape zeros in LMS color space, we need to replace every black pixel
	//max(img_RGB, min_mat, img_RGB); // before converting to LMS
	showMinStd(img_RGB, "initial");
	Mat img_lms;
	transform(img_RGB, img_lms, RGB_to_LMS);
	max(img_lms, min_mat, img_lms); // right before log10
	showMinStd(img_lms, "after convert to LMS");
	// this trick from Jun Yan's code
	// log10(x)=ln(x)/ln(10)
	log(img_lms,img_lms);
	img_lms /= log(10);
	showMinStd(img_lms, "after log10");
	Mat img_lab;
#ifdef SINGLE_MATRIX ˜CV_LOAD_IMAGE_GRAYSCALE
	transform(img_lms, img_lab, LMS_to_lab);
#else
	transform(img_lms, img_lab, LMS_to_lab_1 * LMS_to_lab_2);
#endif
	return img_lab;
}
Mat convertFromlab(Mat input)
{
	Mat img_lms;
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

	Mat img_RGB;
	transform(img_lms, img_RGB, LMS_to_RGB);
	img_RGB.convertTo(img_RGB, CV_8UC1, 255.f);
	Mat img_BGR;
	cvtColor(img_RGB, img_BGR, CV_RGB2BGR);
	return img_BGR;
}
void showMinStd(Mat input, std::string caption)
{
	Mat mean, stdd;
	meanStdDev(input, mean, stdd);
	printf("%s\n", caption.c_str());
	for(int i = 0; i < input.channels(); i++)
	{
		printf("\t%d %f %f\n", i, mean.at<double>(i), stdd.at<double>(i));
	}
}