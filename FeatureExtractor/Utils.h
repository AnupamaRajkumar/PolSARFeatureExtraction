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
		void WriteToFile(int k, double accuracy, int trainSize, int testSize, string& featureName);
		map<string, Vec3f> loadLabelsMetadata();
		Mat_<Vec3f> visualiseLabels(Mat &image, string& imageName);
		void visualiseLabelsF(Mat &image, string& imageName);
		void Visualization(string& fileName, string& imageName, Size size);
		void GetLabelPatchIndex(int sizeOfPatch, Point2i samplePoints, Mat& LabelMap, int& pStart_r, int& pStart_c, int& pEnd_r, int& pEnd_c);
		void DisplayClassName(int finalClass);
		void VisualizationImages(Size size);
		void generateLabelMap(vector<Mat>& label, Mat& labelMap);
		void generateTestLabel(vector<Mat>& label, vector<string>& labelName, Mat& labelMap, int cnt);
		vector<pair<vector<Point2i>, uint>> GetPatchPoints(int patchIdx, Data& data);
		void DivideTrainTestData(Data& data, int fold, int patchIdx);
		void DivideTrainTestData(Data& data, int fold, vector<pair<vector<Point2i>, uint>> patchPoint);
		void DivideTrainTestData2(Data& data, int fold);
		void WriteCoherenceMatValues(vector<pair<vector<float>, unsigned char>>& imgData, string& fileName, bool isApp);
		void WriteCoherenceMatValues(vector<vector<float>>& featureVector, string& fileName, bool isApp);
		void calculateMeanFeatureVector(vector<vector<float>>& featureVector, Mat& outPut);
		void ConvertToCoherenceVector(vector<vector<float>>& result, vector<vector<float>>& coherenceVec);

};
#endif
