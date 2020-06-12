/*Implementing complete processing chain*
*step 1: Read data - from oberpfaffenhofen - rat, label, image
*step 2: Simple feature extraction
*step 3: Train a classifier (KNN/RF/CNN)
*step 4: Apply trained classifier to test data
*step 5: Visualize - PCA/tSNE? and Evaluate data
*/

#include <iostream>
#include <opencv2/opencv.hpp>

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
	int k;
	int numberOfTrainSamples;
	int numberOfTestSamples;
	bool train;
	//Object of class
	Data data;
	KNN knn;
	Feature feature;
	Utils utils;

	/*********Variable Initialization****************/
	k = 5;
	numberOfTrainSamples = 20;
	numberOfTestSamples = 5;

	/*********Function calls****************/
	//load PolSAR data
	data.loadData(argv[1]);
	cout << "Data loaded" << endl;
	//load RGB image
	//Mat RGBImg = data.loadImage(argv[2]);
	//cout << "Image loaded" << endl;
	//load labels
	data.loadLabels(argv[3], data.labelImages, data.labelNames, data.numOfPoints);
	cout << "Labels loaded" << endl;

	
	//Splitting training and testing data for classification
	DivideTrainTestData(numberOfTrainSamples, numberOfTestSamples, data);

	cout << "Training samples:" << data.trainSamples.Samples.size() << endl;
	cout << "Training Labels:" << data.trainSamples.labelName.size() << endl;
	cout << "Testing samples:" << data.testSamples.Samples.size() << endl;
	cout << "Testing Labels:" << data.testSamples.labelName.size() << endl;

	
	/*Computing texture feature extractor*/
	vector<Mat> trainTexture;
	vector<string> trainTextLabels;
	train = true;
	feature.GetTextureFeature(trainTexture, trainTextLabels, data, train);

	/*texture is the training data as it has all the values and labels
	 test data needs to be classified*/
	vector<Mat> testTexture;
	vector<string> testTextLabel;
	train = false;
	feature.GetTextureFeature(testTexture, testTextLabel, data, train);

	knn.KNNTest(trainTexture, trainTextLabels, testTexture, testTextLabel, k);

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
			vector<Point2i> pts;
			if (data.numOfPoints[cnt][pt].y > training_start_idx) {
				//cout << pt << trainCnt << endl;
				/*Ensure that the number of points is less than the max points*/
				if (trainCnt < samplesPerClass) {
					int val = getSafeSamplePoints(data.numOfPoints[cnt][pt], data, samplesPerClass, cnt);
					if (val == 1) {
						data.trainSamples.Samples.push_back(data.numOfPoints[cnt][pt]);
						data.trainSamples.labelName.push_back(data.labelNames[cnt]);
						trainCnt++;
					}
				}
			}
			else
			{
				if (testCnt < numberOfTestSamples) {
					int val = getSafeSamplePoints(data.numOfPoints[cnt][pt], data, samplesPerClass, cnt);
					if (val == 1) {
						data.testSamples.Samples.push_back(data.numOfPoints[cnt][pt]);
						data.testSamples.labelName.push_back(data.labelNames[cnt]);
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





