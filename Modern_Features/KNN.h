#pragma once
#ifndef KNN_H
#define KNN_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;

class KNN {
public:

	void KNNTest(vector<Mat>& trainVal, vector<unsigned char>& trainLabels, vector<Mat>& testVal, 
				 vector<unsigned char>& testLabels, int k, vector<unsigned char>& classResult);
	void OpenCVKNNTest(vector<Mat>& trainVal, vector<unsigned char>& trainLabels, vector<Mat>& testVal,
				 int k, vector<unsigned char>& classResult);
	double Euclidean(Mat& testVal, Mat& trainVal);	
	unsigned char Classify(vector<pair<float, unsigned char>>& distVec, int k);

private:
	vector<Mat> distance;
};


#endif
