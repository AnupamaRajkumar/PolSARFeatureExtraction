#pragma once
#ifndef  SPECKLEFILTER_HPP_
#define  SPECKLEFILTER_HPP_
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

class RefinedLee {
private:
	double NonValidPixelValue = -1.0;

	int filterSize = 9;
	int stride;
	int subWindowSize;
	double sigmaVSqr;

public:
	// Constructor
	// filter size choose from (5, 7, 9, 11)
	RefinedLee(int filter_size, int numLooks = 1) {
		filterSize = filter_size;
		switch (filterSize) {
		case 5:
			subWindowSize = 3;
			stride = 1;
			break;
		case 7:
			subWindowSize = 3;
			stride = 2;
			break;
		case 9:
			subWindowSize = 5;
			stride = 2;
			break;
		case 11:
			subWindowSize = 5;
			stride = 3;
			break;
		default:
			cout << "Unknown filter size: " << filterSize << endl;
			exit(-1);
		}

		double sigmaV = 1.0 / std::sqrt(numLooks);
		sigmaVSqr = sigmaV * sigmaV;
	}

	~RefinedLee() {}

	void filterFullPol(Mat& hh, Mat& vv, Mat& hv);
	void filterFullPol(Mat& hh, Mat& vv, Mat& hv, Mat& span);

	
	//void filterDualPol();

private:
	double computePixelValueUsingLocalStatistics(const Mat& neighborPixelValues, int numSamples);

	double computePixelValueUsingEdgeDetection(const Mat& neighborPixelValues, const Mat& neighborSpanValues);

	int getLocalData(int x, int y, const Mat& src, const Mat& span, Mat& neighborPixelValues, Mat& neighborSpanValues);

	double getLocalMeanValue(const Mat& neighborValues, int numSamples);

	double getLocalVarianceValue(const Mat& neighborValues, int numSamples, double mean);

	

	// Compute mean values for the 3x3 sub-areas in the sliding window 
	void computeSubAreaMeans(const Mat& neighborPixelValues, Mat& subAreaMeans);

	// Compute the gradient in 3x3 subAreaMeans 
	int getDirection(const Mat& subAreaMeans);

	// Get pixel values from the non-edge area indicated by the given direction
	int getNonEdgeAreaPixelValues(const Mat& neighborPixelValues, int d, Mat& pixels);
};


#endif