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


Performance perform;

/***********************************************************************
Calculating euclidean distance between image and label map points
Author : Anupama Rajkumar
Date : 12.06.2020
Description: This function is used to calculate the Euclidean distance 
between the training and test samples
*************************************************************************/

double KNN::Euclidean(Mat& testVal, Mat& trainVal) {
	float distance = 0.0;
	float sum = 0.0;
	//ensure that the dimensions of testVal and trainVal are same
	if ((testVal.rows != trainVal.rows) || (testVal.cols != trainVal.cols))
	{
		cerr << "Matrix dimensions should be same for distance calculation" << endl;
		exit(-1);
	}
	else {
		Mat difference = Mat::zeros(testVal.size(), testVal.type());
		difference =  testVal - trainVal;
		for (int row = 0; row < difference.rows; row++) {
			for (int col = 0; col < difference.cols; col++) {
				sum = sum + pow(difference.at<float>(row, col), 2);
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

unsigned char KNN::Classify(vector<pair<float, unsigned char>>& distVec, int k) {
	int city, field, forest, grassland, street;
	vector<pair<int, unsigned char>> classType;
	classType.reserve(NUMOFCLASSES);
	city = field = forest = grassland = street  = 0;
	
	for (int i = 0; i < k; i++) {	
		/*city*/
		if (distVec[i].second == 1)
		{
			city++;
		}  
		/*Field*/
		else if (distVec[i].second == 2)
		{
			field++;
		}
		/*Forest*/
		else if (distVec[i].second == 3)
		{
			forest++;
		}
		/*Grassland*/
		else if (distVec[i].second == 4)
		{
			grassland;
		}
		/*Street*/
		else if (distVec[i].second == 5)
		{
			street;
		}
	}
	for (int cnt = 0; cnt < (NUMOFCLASSES); cnt++) {
		pair<int, unsigned char> classCnt;
		switch (cnt) {
		case 0:
			classCnt.first = city;
			classCnt.second = 1;
			break;
		case 1:
			classCnt.first = field;
			classCnt.second = 2;
			break;
		case 2:
			classCnt.first = forest;
			classCnt.second = 3;
			break;
		case 3:
			classCnt.first = grassland;
			classCnt.second = 4;
			break;
		case 4:
			classCnt.first = street;
			classCnt.second = 5;
			break;
		default:
			cout << "Invalid classification";
			break;
		}
		classType.push_back(classCnt);
	}
	
	//sort in descending order
	sort(classType.begin(), classType.end(), greater());
	unsigned char classifier;
	classifier = classType[0].second;

	return classifier;
}



/***********************************************************************
Verifying KNN Classifier with random samples
Author : Anupama Rajkumar
Date : 12.06.2020
Description: Classify test points using KNN Classifier
*************************************************************************/

void KNN::KNNTest(vector<Mat>& trainVal, vector<unsigned char>& trainLabels,
				  vector<Mat>& testVal, vector<unsigned char>& testLabels, 
				  vector<unsigned char>& classResult, int k) {
	/*for each sample in the testing data, caculate distance from each training sample */
	Feature f;
	for (int i = 0; i < testVal.size(); i++) {								//for each test sample
		vector<pair<float, unsigned char>> distVec;
		for (int j = 0; j < trainVal.size(); j++) {							//for every training sample
			pair<float, unsigned char> dist;		
			dist.first = this->Euclidean(testVal[i], trainVal[j]);			//calculate euclidean distance
			dist.second = trainLabels[j];
			distVec.push_back(dist);
		}
		//sort the distance in the ascending order
		sort(distVec.begin(), distVec.end());
		//classify for each row the label patch
		unsigned char classVal = this->Classify(distVec, k);
		//cout << "Feature point classified as " << classVal << endl;
		classResult.push_back(classVal);

		if ((i + 1) % 1000 == 0) {
			cout << "1000 samples classified" << endl;
		}
	}	
}


