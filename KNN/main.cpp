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



void DivideTrainTestData(vector<Point2i>& testSamples,
						 int numberOfTrainSamples, int numberOfTestSamples,
						 Data& data);

int main(int argc, char** argv)
{
	cout << "In main!!" << endl;
	/*********Variables Area*************/
	int k;
	int numberOfTrainSamples;
	int numberOfTestSamples;
	//Object of class
	Data data;
	KNN knn;
	Feature feature;
	Utils utils;

	/*********Variable Initialization****************/
	k = 20;
	numberOfTrainSamples = 20000;
	numberOfTestSamples = 4000;

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
	vector<Point2i> testSamples;
	DivideTrainTestData(testSamples, numberOfTrainSamples, numberOfTestSamples, data);

	cout << "Training samples:" << data.trainSamples.Samples.size() << endl;
	cout << "Training samples:" << data.trainSamples.labelName.size() << endl;
	cout << "Testing samples:" << testSamples.size() << endl;

	/*Computing texture feature extractor*/
	vector<Mat> texture;
	vector<vector<string>> textureLabels;
	feature.GetTextureFeature(texture, textureLabels, data);

	//KNN classifier
	//knn.KNNClassifier(data.labelImages, data.labelNames, k, RGBImg);
	
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
void DivideTrainTestData(vector<Point2i>& testSamples,
						 int numberOfTrainSamples, int numberOfTestSamples,
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
			if (data.numOfPoints[cnt][pt].y > training_start_idx) {
				//cout << pt << trainCnt << endl;
				/*Ensure that the number of points is less than the max points*/
				if (trainCnt < samplesPerClass) {
					data.trainSamples.Samples.push_back(data.numOfPoints[cnt][pt]);
					data.trainSamples.labelName.push_back(data.labelNames[cnt]);
					trainCnt++;
				}
			}
			else
			{
				if (testCnt < numberOfTestSamples) {
					testSamples.push_back(data.numOfPoints[cnt][pt]);
					testCnt++;
				}				
			}
		}
	}

}


