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

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	cout << "In main!!" << endl;
	/*********Variables Area*************/
	int k;
	vector<Mat> labelImages;
	vector<string> labelNames;
	vector<vector<Point2i>> numOfPoints;

	/*********Variable Initialization****************/
	k = 20;
	labelImages.reserve(NUMOFCLASSES);
	labelNames.reserve(NUMOFCLASSES);
	numOfPoints.reserve(NUMOFCLASSES);
	
	//Object of class
	Data data;
	KNN knn;

	//load RAT
	//data.loadData(argv[1]);
	cout << "Data loaded" << endl;
	//load RGB image
	Mat RGBImg = data.loadImage(argv[2]);
	cout << "Image loaded" << endl;
	//load labels
	data.loadLabels(argv[3], labelImages, labelNames, numOfPoints);
	cout << "Labels loaded" << endl;

	//KNN classifier
	knn.KNNClassifier(labelImages, labelNames, k, RGBImg);
	
	waitKey(0);
	return 0;	
}


