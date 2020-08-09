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
Description: This function is used to calculate the Euclidean distance 
between the training and test samples
Input 1 : test points
Input 2 : training points
Output  : Calculated euclidean distance
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




/***************************************************************************************
Verifying KNN Classifier Result
Author : Anupama Rajkumar
Description : This function counts the number of classes in k neighborhood
Based on which class has the highest count, appropriate class is returned
Input 1 : The sorted distance between test and training points
Input 2 : Hyperparameter k
Output	: Classified point value
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
KNN Classifier
Author : Anupama Rajkumar
Description: Classify test points using KNN Classifier
Input 1 : Training values
Input 2 : Training labels
Input 3 : Test values
Input 4 : Test labels
Input 5 : Hyperparameter k
Output  : Classification result
*************************************************************************/

void KNN::KNNTest(vector<Mat>& trainVal, vector<unsigned char>& trainLabels,
				  vector<Mat>& testVal, vector<unsigned char>& testLabels, 
				  int k , vector<unsigned char>& classResult) {
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


/***********************************************************************
OpenCV KNN Classifier
Author : Anupama Rajkumar
Description: Classify test points using KNN Classifier by OpenCV
Input 1 : Training values
Input 2 : Training labels
Input 3 : Test values
Input 4 : Test labels
Input 5 : Hyperparameter k
Output  : Classification result
*************************************************************************/
void KNN::OpenCVKNNTest(vector<Mat>& trainVal, vector<unsigned char>& trainLabels, vector<Mat>& testVal, 
						int k, vector<unsigned char>& classResult) {
	cv::Mat trainData, trainDataLabel;
	vconcat(trainVal, trainData);
	vconcat(trainLabels, trainDataLabel);
	trainDataLabel.convertTo(trainDataLabel, CV_32SC1);
	trainData.convertTo(trainData, CV_32FC1);
	cv::Ptr<cv::ml::TrainData> cv_data = cv::ml::TrainData::create(trainData, 0, trainDataLabel);
	cv::Ptr<cv::ml::KNearest>  knn(cv::ml::KNearest::create());
	knn->setDefaultK(k);
	knn->setIsClassifier(true);
	knn->train(cv_data);
	int cnt = 0;
	for (auto& x_test : testVal) {
		cnt++;
		x_test.convertTo(x_test, CV_32FC1);
		auto knn_result = knn->predict(x_test);
		classResult.push_back(unsigned char(knn_result));
		if ((cnt + 1) % 50000 == 0) {
			cout << cnt << " points classified" << endl;
		}
	}
}



