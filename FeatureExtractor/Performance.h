#pragma once
#ifndef PERFORMANCE_H
#define PERFORMANCE_H


#include <iostream>
#include <opencv2/opencv.hpp>
#include "KNN.h"

using namespace std;
using namespace cv;

class Performance {
public:
	double calculatePredictionAccuracy(vector<unsigned char>& classResult, vector<unsigned char>& testLabels, vector<pair<vector<Point2i>, uint>>& patchPoint);
	double calculatePredictionAccuracy(vector<unsigned char>& classResult, vector<unsigned char>& testLabels);
};



#endif
