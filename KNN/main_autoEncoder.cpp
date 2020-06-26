/*Implementing complete processing chain*
*step 1: Read data - from oberpfaffenhofen - rat, label, image
*step 2: Simple feature extraction
*step 3: Train a classifier (KNN/RF/CNN)
*step 4: Apply trained classifier to test data
*step 5: Visualize - PCA/tSNE? and Evaluate data
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <algorithm>

#include "Data.h"
#include "KNN.h"
#include "Feature.h"
#include "Utils.h"
#include "mp.hpp"


using namespace std;
using namespace cv;



void DivideTrainTestData(int numberOfTrainSamples, 
						 int numberOfTestSamples,
						 Data& data);


int getSafeSamplePoints(Point2i samplePoint, Data& data, int samplesPerClass, int cnt);


int main(int argc, char** argv)
{
	cout << "In main!!" << endl;
	/*********Variables Area*************/
	int k, kSize;
	int numberOfTrainSamples;
	int numberOfTestSamples;
	bool train;
	//Object of class
	Data data;
	KNN knn;
	Feature feature;
	Utils utils;
	string featureName;

	/*********Variable Initialization****************/
	k = 1;
	kSize = 3;
	numberOfTrainSamples = 5;
	numberOfTestSamples = 1;

	/*********Function calls****************/
	//load PolSAR data
	data.loadData(argv[1]);
	cout << "Data loaded" << endl;

	/*	1 -> City
		2 -> Field
		3 -> Forest
		4 -> Grassland
		5 -> Street		*/
	
	data.loadLabels(argv[3], data.labelImages, data.labelNames, data.numOfPoints);
	cout << "Labels loaded" << endl;
	
	/*calculate feature vector of all the images*/
	vector<Mat> result;
	feature.GetCoherencyFeatures(data, result);

	waitKey(0);
	return 0;	
}



/************************************************************
Dividing the data samples into training and test samples
Take some training samples for each class and same for 
test samples
Date: 11.06.2020
Author: Anupama Rajkumar
*************************************************************/
void DivideTrainTestData(int numberOfTrainSamples, 
						 int numberOfTestSamples,
						 Data& data) {
	int training_start_idx = int(data.labelImages[0].cols / 5);

	int trainCnt, testCnt;
	//random samples generator
	std::random_device rd;										   // obtain a random number from hardware
	std::mt19937 eng(rd());										   // seed the generator
	std::uniform_int_distribution<> distrX(0, data.labelImages[0].rows);		   // define the range
	std::uniform_int_distribution<> distrY(0, data.labelImages[0].cols);

	/*The idea is to get a balanced division between all the classes. 
	5 classes with equal number of points. Also, the first 1/5th region is 
	reserved for testing data set and from remaining area training samples are taken*/
	int samplesPerClass = int(numberOfTrainSamples / NUMOFCLASSES);
	/*for each class*/
	for (int cnt = 0; cnt < data.numOfPoints.size(); cnt++) {										
		trainCnt = 0;
		testCnt = 0;
		/*for each point in each class*/
		for (int pt = 0; pt < data.numOfPoints[cnt].size(); pt++) {
			int x = distrX(eng);
			int y = distrY(eng);
			Point2i newSample(data.numOfPoints[cnt][pt].x, data.numOfPoints[cnt][pt].y);
			if (newSample.y > training_start_idx) {
				//cout << "newsample:" << newSample.x << "x" << newSample.y << endl;
				//cout << pt << trainCnt << endl;
				/*Ensure that the number of points is less than the max points*/
				if (trainCnt < samplesPerClass) {
					int val = getSafeSamplePoints(newSample, data, samplesPerClass, cnt);
					if (val == 1) {						
						data.trainSamples.Samples.push_back(newSample);
						data.trainSamples.labelName.push_back(cnt+1);		//data.labelNames[cnt]				
						trainCnt++;
					}
				}
			}
			else
			{
				if (testCnt < numberOfTestSamples) {
					int val = getSafeSamplePoints(newSample, data, samplesPerClass, cnt);
					if (val == 1) {
						data.testSamples.Samples.push_back(newSample);
						data.testSamples.labelName.push_back(cnt+1);			//data.labelNames[cnt]
						testCnt++;
					}
				}				
			}
		}
	}
}


int getSafeSamplePoints(Point2i samplePoint, Data& data, int samplesPerClass, int cnt) {

	Point2i new_ind;
	int j_min = samplePoint.x - int(data.sizeOfPatch / 2); 
	int j_max = samplePoint.x + int(data.sizeOfPatch / 2);
	int i_min = samplePoint.y - int(data.sizeOfPatch / 2);
	int i_max = samplePoint.y + int(data.sizeOfPatch / 2);
	// get rid of the points on the borders
	if (i_max < data.labelImages[cnt].cols && j_max < data.labelImages[cnt].rows && i_min >= 0 && j_min >= 0) {
		// get rid of points which are half patch size away from the mask zero area
			return 1;				
	}
	else {
		return 0;
	}
}





