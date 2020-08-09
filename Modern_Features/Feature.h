#pragma once
#ifndef FEATURE_H
#define FEATURE_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "Data.h"

//#if 0
using namespace std;
using namespace cv;


class Feature {
public:


	/************Variables******************/
	string featureName;
	/************Variables******************/

	void getLexiBasis(const Mat& hh, const Mat& vv, const Mat& hv, vector<Mat>& lexi);
	void getPauliBasis(const Mat& hh, const Mat& vv, const Mat& hv, vector<Mat>& pauli);
	Mat logTransform(const Mat& in);  //intensity in dB
	vector<float> logTransform(vector<float>& in);
	Mat getComplexAmpl(const Mat& in);
	void vec2mat(const vector<Mat>& basis, int winSize, vector<Mat>& mat);
	void GetCoherencyFeatures(Data data, vector<vector<float>>& result, vector<unsigned char>& classValue);
	void GetCoherencyMat(vector<Mat>& pauli, int winSize, vector<Mat>& coherencyMat);
};

#endif
//#endif


