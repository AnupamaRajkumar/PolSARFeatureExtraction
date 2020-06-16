#include "KNN.h"
#include "Data.h"
#include "Utils.h"
#include "Performance.h"
#include "Feature.h"

#include <iostream>
#include <math.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <cmath>

using namespace std;
using namespace cv;

Utils utils;
Performance perform;

/***********************************************************************
Calculating euclidean distance between image and label map points
Author : Anupama Rajkumar
Date : 12.06.2020
Description: This function is used to calculate the Euclidean distance 
between the training and test samples
*************************************************************************/

double KNN::Euclidean(Mat& testVal, Mat& trainVal) {
	double distance = 0.0;
	double sum = 0.0;
	//ensure that the dimensions of testVal and trainVal are same
	if ((testVal.rows != trainVal.rows) || (testVal.cols != trainVal.cols))
	{
		cerr << "Matrix dimensions should be same for distance calculation" << endl;
		exit(-1);
	}
	else {
		int row = testVal.rows;
		int col = testVal.cols;
		for (int i = 0; i < row; i++) {
			for (int j = 0; j < col; j++) {
				double diff = testVal.at<float>(i, j) - trainVal.at<float>(i, j);
				sum = sum + pow(diff, 2);
				distance = sqrt(sum);
			}
		}
	}

	//cout << "distance = " << distance;
	return distance;
}




/***********************************************************************
Verifying KNN Classifier with random samples - Overloaded Function
Author : Anupama Rajkumar
Date : 28.05.2020
Description : This function counts the number of classes in k neighborhood
Based on which class has the highest count, appropriate class is returned
*************************************************************************/

string KNN::Classify(vector<pair<double, string>>& distVec, int k) {
	int city, field, forest, grassland, street;
	vector<pair<int, string>> classType;
	classType.reserve(NUMOFCLASSES);
	city = field = forest = grassland = street  = 0;
	
	for (int i = 0; i < k; i++) {	
		if (distVec[i].second == "city_inter")
		{
			city++;
		}  
		else if (distVec[i].second == "field_inter")
		{
			field++;
		}
		else if (distVec[i].second == "forest_inter")
		{
			forest++;
		}
		else if (distVec[i].second == "grassland_inter")
		{
			grassland;
		}
		else if (distVec[i].second == "street_inter")
		{
			street;
		}
	}
	for (int cnt = 0; cnt < (NUMOFCLASSES); cnt++) {
		pair<int, string> classCnt;
		switch (cnt) {
		case 0:
			classCnt.first = city;
			classCnt.second = "city_inter";
			break;
		case 1:
			classCnt.first = field;
			classCnt.second = "field_inter";
			break;
		case 2:
			classCnt.first = forest;
			classCnt.second = "forest_inter";
			break;
		case 3:
			classCnt.first = grassland;
			classCnt.second = "grassland_inter";
			break;
		case 4:
			classCnt.first = street;
			classCnt.second = "street_inter";
			break;
		default:
			cout << "Invalid classification";
			break;
		}
		classType.push_back(classCnt);
	}
	
	//sort in descending order
	sort(classType.begin(), classType.end(), greater());
	string classifier;
	classifier = classType[0].second;

	return classifier;
}



/***********************************************************************
Verifying KNN Classifier with random samples
Author : Anupama Rajkumar
Date : 12.06.2020
Description: Classify test points using KNN Classifier
*************************************************************************/

void KNN::KNNTest(vector<Mat>& trainVal, vector<string>& trainLabels, 
				  vector<Mat>& testVal, vector<string>& testLabels, 
				  int k, string& featureName) {
	/*for each sample in the testing data, caculate distance from each training sample */
	vector<string> classResult;
	Feature f;
	for (int i = 0; i < testVal.size(); i++) {								//for each test sample
		vector<pair<double, string>> distVec;
		for (int j = 0; j < trainVal.size(); j++) {							//for every training sample
			pair<double, string> dist;		
			dist.first = this->Euclidean(testVal[i], trainVal[j]);			//calculate euclidean distance
			dist.second = trainLabels[j];
			distVec.push_back(dist);
		}
		//sort the distance in the ascending order
		sort(distVec.begin(), distVec.end());
		//classify for each row the label patch
		string classVal = this->Classify(distVec, k);
		//cout << "Feature point classified as " << classVal << endl;
		classResult.push_back(classVal);
	}	
	double accuracy = perform.calculatePredictionAccuracy(classResult, testLabels);
	cout << "Accuracy: " << accuracy << endl;
	//log the calculated accuracy
	cout << "feature name:" << featureName << endl;
	utils.WriteToFile(k, accuracy, trainVal.size(), testVal.size(), featureName);
}


