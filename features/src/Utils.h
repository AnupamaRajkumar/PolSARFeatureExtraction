#pragma once
#ifndef UTILS_H
#define UTILS_H

#include<iostream>
#include<opencv2/opencv.hpp>
#include "Geotiff.hpp"

using namespace std;
using namespace cv;

namespace Utils {
	    // by Anupama Rajkumar
		void WriteToFile(Mat& labelMap, string& fileName);
		map<string, Vec3f> loadLabelsMetadata();
		Mat_<Vec3f> visualiseLabels(Mat &image, string& imageName);
		void GetPatchIndex(int sizeOfPatch, Point2i& samplePoint, const Mat& LabelMap, int& min_col, int& min_row, int& max_col, int& max_row);
		void Visualization(string& fileName, string& imageName, Size size);

		double calculatePredictionAccuracy(const vector<unsigned char>& classResult, const vector<unsigned char>& testLabels);
		Mat generateLabelMap(const vector<Mat>& masks);

		// by Jun Xiang

		//check if dataset exist in hdf5 file
		bool checkExistInHDF(const String& filename, const String& parent_name, const vector<string>& dataset_name,int filterSize,int patchSize);
		bool checkExistInHDF(const String& filename, const String& parent_name, const string& dataset_name);

		// delete dataset from hdf5 file
		void deleteDataFromHDF(const String& filename, const String& parent_name, const String& dataset_name);
		void deleteDataFromHDF(const String& filename, const String& parent_name, const vector<String>& dataset_name, int filterSize, int patchSize);
		
		// eg: filename = "ober.h5", parent_name = "/filtered_data", dataset_name = "/hh_filterSize_5"
		void writeDataToHDF(const String& filename, const String& parent_name, const String& dataset_name, const Mat& data);
		void writeDataToHDF(const String& filename, const String& parent_name, const vector<String>& dataset_name, const vector<Mat>& data,int filterSize =0, int patchSize =0);

		// eg: filename = "ober.h5" ,parent_name = "/filtered_data", dataset_name = "/hh_filterSize_5"
		void readDataFromHDF(const String& filename, const String& parent_name, const String& dataset_name, Mat& data);
		void readDataFromHDF(const String& filename, const String& parent_name, const vector<String>& dataset_name,  vector<Mat>& data, int filterSize =0, int patchSize =0);

		//write attribute to the root group
		void writeAttrToHDF(const String& filename, const String& attribute_name, const int &attribute_value);
		void writeAttrToHDF(const String& filename, const String& attribute_name, const string &attribute_value);
		void readAttrFromHDF(const String& filename, const String& attribute_name, int& attribute_value);
		void readAttrFromHDF(const String& filename, const String& attribute_name, string& attribute_value);

		//insert data
		bool insertDataToHDF(const String& filename, const String& parent_name, const String& dataset_name, const Mat& data);
		bool insertDataToHDF(const String& filename, const String& parent_name, const vector<string>& dataset_name, const vector<Mat>& data, int filterSize, int patchSize);

		Mat readTiff(string filepath);

		void getSafeSamplePoints(const Mat& labelMap, const int & samplePointNum, const int& sampleSize, vector<Point>& pts);

		void shuffleDataSet(vector<Mat>& data, vector<unsigned char>& data_label);

		void DivideTrainTestData(const vector<Mat>& data, const vector<unsigned char>& data_label, int percentOfTrain,
			vector<Mat>& train_img, vector<unsigned char>& train_label, vector<Mat>& test_img, vector<unsigned char>& test_label);

		Mat convertDataToPNG(const Mat& src);

		Mat getConfusionMatrix(const map<unsigned char, string>& className, vector<unsigned char>& classResult, vector<unsigned char>& testLabels);

};
#endif
