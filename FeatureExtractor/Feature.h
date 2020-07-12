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
	//int sizeOfPatch = 10;
	string featureName;
	/************Variables******************/

	void lexi2pauli(vector<Mat>& lexi, vector<Mat>& pauli);
	void getLexiBasis(const Mat& hh, const Mat& vv, const Mat& hv, vector<Mat>& lexi);
	void getPauliBasis(const Mat& hh, const Mat& vv, const Mat& hv, vector<Mat>& pauli);
	//void getPauliBasis(vector<double>& hh, vector<double>& vv, vector<double>& hv, vector<vector<double>>& pauli);
	void GetTextureFeature(vector<Mat>& features, vector<unsigned char>& classValue, Data data, bool flag);
	Mat logTransform(const Mat& in);  //intensity in dB
	vector<float> logTransform(vector<float>& in);
	Mat getComplexAmpl(const Mat& in);
	void vec2mat(const vector<Mat>& basis, vector<Mat>& mat, int winSize);
	void GetCoherencyFeatures(Data data, vector<vector<float>>& result, vector<unsigned char>& classValue);
	void GetCoherencyMat(vector<Mat>& pauli, vector<Mat>& coherencyMat, int winSize);
	//void GetCoherencyVec(vector<vector<double>>& pauliVec, vector<vector<double>>& coherencyVec, int winSize);
};

#endif
//#endif


