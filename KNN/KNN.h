#pragma once
#ifndef KNN_H
#define KNN_H

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class KNN {
public:

	void KNNTest(vector<Mat>& trainVal, vector<string>& trainLabels, vector<Mat>& testVal, vector<string>& testLabels, int k, string& featureName);
	double Euclidean(Mat& testVal, Mat& trainVal);	
	string Classify(vector<pair<double, string>>& distVec, int k);

private:
	vector<Mat> distance;
};


#endif
