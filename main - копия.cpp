#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string.h>
#include <vector>
#include <math.h>

using namespace cv;

#define WND_NAME_RES "Result Picture"

Mat RGB_to_LMS = (Mat_<double>(3,3) <<	0.3811, 0.5783, 0.0402,
										0.1967, 0.7244, 0.0782,
										0.0241, 0.1288, 0.8444);
Mat LMS_to_RGB = (Mat_<double>(3,3) <<	4.4679, -3.5873, 0.1193,
										-1.2186, 2.3809, -0.1624,
										0.0497, -0.2439, 1.2045);
Mat LMS_to_lab_1 = (Mat_<double>(3,3) << 1/sqrt(3), 0, 0,
										0, 1/sqrt(6), 0,
										0, 0, 1/sqrt(2));
Mat LMS_to_lab_2 = (Mat_<double>(3,3) << 1, 1, 1,
											1, 1, -2,
											1, -1, 0);
Mat LMS_to_lab, lab_to_LMS;
struct ct_image
{
	std::string source;
	std::string target;
	std::string result;
};

bool makeCT(ct_image images);

Mat convertTolab(Mat input);
Mat convertFromlab(Mat input);
Mat mult(Mat mat1, Mat mat2);
Mat _transform(Mat mat, Mat core);

int main()
{
	//transpose(RGB_to_LMS, RGB_to_LMS);
	//transpose(LMS_to_RGB, LMS_to_RGB);
	//GEMM_1_T GEMM_2_T GEMM_3_T
	//gemm(LMS_to_lab_1, LMS_to_lab_2, 1, Mat(), 0, LMS_to_lab, GEMM_2_T);
	//gemm(LMS_to_lab_2, LMS_to_lab_1, 1, Mat(), 0, lab_to_LMS, GEMM_1_T);
	transpose(LMS_to_lab_1, LMS_to_lab_1);
	transpose(LMS_to_lab_2, LMS_to_lab_2);
	ct_image images_1 = {"pic_1.jpg", "pic_2.jpg", "pic_1_2(1).jpg"};
	if(makeCT(images_1))
	{
		Mat res_pic = imread(images_1.result);
		imshow(WND_NAME_RES, res_pic);
		waitKey(0);
	}
	return 0;
}
bool makeCT(ct_image images)
{
	Mat imgs = imread(images.source);
	Mat imgt = imread(images.target);
	Mat imgs_lab = convertTolab(imgs);
	Mat imgt_lab = convertTolab(imgt);
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
	result = convertFromlab(result);
	imwrite(images.result, result);
	return true;
}
Mat convertTolab(Mat input)
{
	Mat img;
	input.convertTo(img,CV_32FC1);
	Mat img_lms;
	img_lms = _transform(img, RGB_to_LMS);
	MatIterator_<Vec3f> iter = img_lms.begin<Vec3f>();
	for(; iter != img_lms.end<Vec3f>(); iter++)
	{
		(*iter)[0] = log10((*iter)[0]);
		(*iter)[1] = log10((*iter)[1]);
		(*iter)[2] = log10((*iter)[2]);
	}
	Mat img_lab;
	img_lab = _transform(img_lms, LMS_to_lab_2);
	img_lab = _transform(img_lab, LMS_to_lab_1);
	return img_lab;
}
Mat convertFromlab(Mat input)
{
	Mat img_lms;
	transpose(LMS_to_lab_1, LMS_to_lab_1);
	transpose(LMS_to_lab_2, LMS_to_lab_2);
	img_lms = _transform(input, LMS_to_lab_1);
	img_lms = _transform(img_lms, LMS_to_lab_2);
	MatIterator_<Vec3f> iter = img_lms.begin<Vec3f>();
	for(; iter != img_lms.end<Vec3f>(); iter++)
	{
		(*iter)[0] = pow((*iter)[0], 10);
		(*iter)[1] = pow((*iter)[1], 10);
		(*iter)[2] = pow((*iter)[2], 10);
	}
	Mat img_RGB;
	img_RGB = _transform(img_lms, LMS_to_RGB);
	img_RGB.convertTo(img_RGB,CV_8UC1);
	return img_RGB;
}
Mat mult(Mat mat1, Mat mat2)
{
	Mat res = Mat::zeros(mat1.rows, mat2.cols, CV_32FC3);
	for(int i = 0; i < mat1.rows; i++)
	for(int j = 0; j < mat2.cols; j++)
	{
		for(int k = 0; k < mat1.cols; k++)
			res.at<float>(i, j) += mat1.at<float>(i, k) * mat2.at<float>(k, j);
	}
	return res;
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