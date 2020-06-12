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
	int sizeOfPatch = 64;
	/************Variables******************/

	void lexi2pauli(vector<Mat>& lexi, vector<Mat>& pauli);
	void GetTextureFeature(vector<Mat>& features, vector<vector<string>>& classValue, Data data);
	Mat logTransform(const Mat& in);  //intensity in dB
	Mat getComplexAmpl(const Mat& in);
};

#endif
//#endif


