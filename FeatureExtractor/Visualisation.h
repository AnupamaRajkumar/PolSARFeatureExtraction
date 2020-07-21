#pragma once
#ifndef __VISUALISATION__
#define __VISUALISATION__

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

class Visual {
public:
	void CalculateMeanFeatureVector(vector<vector<float>>& featureVector, Mat& outPut);
	void GenerateFeatureMap(vector<vector<float>>& m_featureVector);
};

#endif // !__VISUALISATION__
