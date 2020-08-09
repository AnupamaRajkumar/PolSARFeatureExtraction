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
	void ContrastCorrection(vector<vector<float>>& featureVector, int cnt, Mat& outPut);
	void GenerateFeatureMap(vector<vector<float>>& m_featureVector, string& imagePrefix);
};

#endif // !__VISUALISATION__
