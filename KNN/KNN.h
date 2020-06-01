#pragma once
#ifndef KNN_H
#define KNN_H

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class KNN {
public:

	void KNNClassifier(vector<Mat>& label, vector<string>& labelName, int k, Mat& RGBImg);
	void KNNTest(vector<Point2i>& samplesImg, vector<Point2i>& samplesLab, Mat& RGBImg, Mat& LabelMap, int k);
	void KNNTrain(Mat& RGBImg, Mat& LabelMap, int k);
	double Euclidean(int imgX, int imgY, int labX, int labY);
	void generateLabelMap(vector<Mat>& label, vector<string>& labelName, Mat& labelMap);
	int Classify(vector<pair<double, int>>& distVec, int k);
	int Classify(vector<int> classResult);

private:
	vector<Mat> distance;
};


#endif
