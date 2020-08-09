#ifndef  CVFEATURES_HPP_
#define  CVFEATURES_HPP_
#include <opencv2/opencv.hpp>
#include "glcm.hpp"
#include "elbp.hpp"
#include "../MPEG-7/Feature.h"
#include "morph.hpp"

namespace cvFeatures {

	// get LBP feature for grayscale img 
	cv::Mat GetLBP(const cv::Mat &src, int radius =1, int neighbors =8, int histsize = 32);

	// get GLCM features(energy, contrast, homogenity and entropy) for grayscale img 
	cv::Mat GetGLCM(const cv::Mat &src,int winsize = 7, GrayLevel level = GrayLevel::GRAY_16, int histsize =32);

	// get color features for BGR img
	cv::Mat GetMPEG7DCD(const cv::Mat& src, int numOfColor =1);
	cv::Mat GetMPEG7CSD(const cv::Mat& src, int size =32);
	cv::Mat GetHSV(const cv::Mat& src, int size= 32);

	// get the convex/concave/flat area of grayscale img by the derivative of opening-by reconstruction and closing-by reconstruction from original image
	cv::Mat GetMP(const cv::Mat& src, const std::array<int, 3>& morph_size = { 1,3,5 });

	// Compute median, min, max, mean, std for single channel mat
	cv::Mat GetStatistic(const cv::Mat& src);
	
	cv::Mat GetHistOfMaskArea(const cv::Mat& src, const cv::Mat& mask, int minVal=0, int maxVal=255, int histSize =32, bool normed = true);
};

#endif
