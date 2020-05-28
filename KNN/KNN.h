#pragma once
#ifndef KNN_H
#define KNN_H

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class KNN {
public:
	/*vector<Mat> label : labels (training data)
	  int cnt : number of label classes
	  int k   : number of k nearest neighbor
	  Mat label: RGB image*/
	void KNNClassifier(vector<Mat>& label, vector<string>& labelName, int k, Mat& RGBImg);

private:
	//vector<vector<pair<double, string>>> distance;
	vector<Mat> distance;
};


#endif
