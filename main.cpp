#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string.h>
#include <vector>
#include <math.h>
#include <conio.h>

using namespace cv;

#define WND_NAME_RES "Result Picture"
#define WND_NAME_SOURCE "Source Picture"

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
float M_rgb2lms[]={
	0.3811f, 0.5783f, 0.0402f,
	0.1967f, 0.7244f, 0.0782f,
	0.0241f, 0.1228f, 0.8444f
};

struct ct_image
{
	std::string source;
	std::string target;
	std::string result;
};

ct_image images[] = {
	{"images/1/img_1.jpg", "images/1/img_2.jpg", "images/1/img_1_2_cv.jpg"},
	{"images/2/img_3.jpg", "images/2/img_4.jpg", "images/2/img_3_4_cv.jpg"}, // result image becomes black
	{"images/3/pic_1.jpg", "images/3/pic_2.jpg", "images/3/pic_1_2_cv.jpg"},
	{"images/4/test_1.png", "images/4/test_1.png", "images/4/test_1_1.png"},
	{"images/5/test_2.png", "images/5/test_2.png", "images/5/test_2_2.png"}};

bool makeCT(ct_image images);
bool makeCTCIE(ct_image images);

Mat convertTolab(Mat input);
Mat convertFromlab(Mat input);
Mat _transform(Mat mat, Mat core);

void showMinStd(Mat input, std::string caption);

//#define SINGLE_MATRIX
#define FROM_FLOAT

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
	//transpose(RGB_to_LMS, RGB_to_LMS);
	//transpose(LMS_to_RGB, LMS_to_RGB);
	//transpose(LMS_to_lab, LMS_to_lab);
	//transpose(lab_to_LMS, lab_to_LMS);
	//transpose(LMS_to_lab_2, LMS_to_lab_2);

	unsigned img_pack = 3;

	/*Mat temp = imread(images[img_pack].target);
	imshow(WND_NAME_RES, convertFromlab(convertTolab(temp)));
	imshow(WND_NAME_SOURCE, temp);
	waitKey(0);*/
/*	if(makeCT(images[img_pack]))
	{
		Mat res_pic = imread(images[img_pack].result);
		imshow(WND_NAME_RES, res_pic);
		waitKey(0);
	}*/
// FROM HERE
	Mat image = imread(images[img_pack].source);
	Mat img_lms;
#ifdef FROM_FLOAT
	Mat tr_rgb2lms(3,3,CV_32FC1,M_rgb2lms);
	transform(image, img_lms, tr_rgb2lms); // here
#else
	transform(image, img_lms, RGB_to_LMS); // and here
#endif
	showMinStd(img_lms, "after convert to LMS");

	_getch();
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

	std::vector<Mat> imgs_lab_channels, imgt_lab_channels;
	split(imgs_lab, imgs_lab_channels);
	split(imgt_lab, imgt_lab_channels);

	for(int i = 0; i < 3; i++)
	{
		imgs_lab_channels[i] = (imgs_lab_channels[i] - means.at<double>(i)) *
		stddt.at<double>(i)/stdds.at<double>(i) + meant.at<double>(i);
	}
	showMinStd(imgs_lab, "in lab source");
	showMinStd(imgt_lab, "in lab target");
	Mat result;
	merge(imgs_lab_channels, result);
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
	img_RGB.convertTo(img_RGB, CV_32FC1, 1/255.f);
	showMinStd(img_RGB, "initial");
	Mat img_lms;
#ifdef FROM_FLOAT
	Mat tr_rgb2lms(3,3,CV_32FC1,M_rgb2lms);
	transform(img_RGB, img_lms, tr_rgb2lms); // here
#else
	transform(img_RGB, img_lms, RGB_to_LMS); // and here
#endif
	showMinStd(img_lms, "after convert to LMS");
	/*MatIterator_<Vec3f> iter = img_lms.begin<Vec3f>();
	for(; iter != img_lms.end<Vec3f>(); iter++)
	{
		(*iter)[0] = log10((*iter)[0]);
		(*iter)[1] = log10((*iter)[1]);
		(*iter)[2] = log10((*iter)[2]);
	}*/
	// this trick from Jun Yan's code
	// log10(x)=ln(x)/ln(10)
	log(img_lms,img_lms);
	img_lms /= log(10);

	showMinStd(img_lms, "after log10");
	Mat img_lab;
#ifdef SINGLE_MATRIX
	transform(img_lms, img_lab, LMS_to_lab);
#else
	//transform(img_lms, img_lab, LMS_to_lab_1);
	//transform(img_lab, img_lab, LMS_to_lab_2);
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
	//transform(input, img_lms, LMS_to_lab_2); // again limits don't do this at home
	//transform(img_lms, img_lms, LMS_to_lab_1_);
	transform(input, img_lms, LMS_to_lab_2 * LMS_to_lab_1);
#endif
	
	/*MatIterator_<Vec3f> iter = img_lms.begin<Vec3f>();
	for(; iter != img_lms.end<Vec3f>(); iter++)
	{
		(*iter)[0] = pow(10, (*iter)[0]);
		(*iter)[1] = pow(10, (*iter)[1]);
		(*iter)[2] = pow(10, (*iter)[2]);
	}*/
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
Mat _transform(Mat mat, Mat core)
{
	Mat res = mat.clone();
	MatIterator_<Vec3f> iter = res.begin<Vec3f>();
	for(; iter != res.end<Vec3f>(); iter++)
	{
		(*iter)[0] = core.at<float>(0, 0) * (*iter)[0] +  core.at<float>(0, 1) * (*iter)[1] +  core.at<float>(0, 2) * (*iter)[2];
		(*iter)[1] = core.at<float>(1, 0) * (*iter)[0] +  core.at<float>(1, 1) * (*iter)[1] +  core.at<float>(1, 2) * (*iter)[2];
		(*iter)[1] = core.at<float>(2, 0) * (*iter)[0] +  core.at<float>(2, 1) * (*iter)[1] +  core.at<float>(2, 2) * (*iter)[2];
	}
	return res;
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