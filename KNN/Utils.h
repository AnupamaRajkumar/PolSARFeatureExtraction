#pragma once
#ifndef UTILS_H
#define UTILS_H

#include<iostream>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Utils {
	public:
		void WriteToFile(Mat& labelMap, string& fileName);
		map<string, Vec3f> loadLabelsMetadata();
		Mat_<Vec3f> visualiseLabels(Mat &image, string& imageName);
		void Visualization(string& fileName, string& imageName, Size size);
		void GetLabelPatchIndex(int sizeOfPatch, Point2i samplePoints, Mat& LabelMap, int& pStart_r, int& pStart_c, int& pEnd_r, int& pEnd_c);
		void DisplayClassName(int finalClass);

};
#endif
