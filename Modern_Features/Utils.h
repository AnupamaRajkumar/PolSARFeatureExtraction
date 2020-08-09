#pragma once
#ifndef UTILS_H
#define UTILS_H

#include<iostream>
#include<opencv2/opencv.hpp>
#include "Data.h"

using namespace std;
using namespace cv;

class Utils {
	public:
		map<string, Vec3f> loadLabelsMetadata();
		Mat_<Vec3f> visualiseLabels(Mat &image, string& imageName);
		void generateLabelMap(vector<Mat>& label, Mat& labelMap);
		vector<pair<vector<Point2i>, uint>> GetPatchPoints(int patchIdx, Data& data);
		void DivideTrainTestData(Data& data, int fold, int patchIdx);
		void DivideTrainTestData(Data& data, int fold, vector<pair<vector<Point2i>, uint>> patchPoint);
		void WriteCoherenceMatValues(vector<pair<vector<float>, unsigned char>>& imgData, string& fileName, bool isApp);
		void WriteCoherenceMatValues(vector<vector<float>>& featureVector, string& fileName, bool isApp);
		void ConvertToCoherenceVector(vector<vector<float>>& result, vector<vector<float>>& coherenceVec);

};
#endif
