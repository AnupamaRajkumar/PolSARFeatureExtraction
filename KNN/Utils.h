#pragma once
#ifndef UTILS_H
#define UTILS_H

#include<iostream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Utils {
	public:
		void WriteToFile(int k, double accuracy, int trainSize, int testSize, string& featureName);
		map<string, Vec3f> loadLabelsMetadata();
		Mat_<Vec3f> visualiseLabels(Mat &image, string& imageName);
		void Visualization(string& fileName, string& imageName, Size size);
		void Visualization(Mat& inputImg, Mat& outputImg);
		void GetLabelPatchIndex(int sizeOfPatch, Point2i samplePoints, Mat& LabelMap, int& pStart_r, int& pStart_c, int& pEnd_r, int& pEnd_c);
		void DisplayClassName(int finalClass);
		void VisualizationImages(Size size);
		void generateLabelMap(vector<Mat>& label, Mat& labelMap);
		void generateTestLabel(vector<Mat>& label, vector<string>& labelName, Mat& labelMap, int cnt);
		void getAverageFilter(vector<Mat>& trainTexture, vector<Mat>& filtTrainText, int kSize);

};
#endif
