#include "KNN.h"
#include "Data.h"
#include "Utils.h"

#include <iostream>
#include <math.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

using namespace std;
using namespace cv;

Utils utils;

/***********************************************************************
Calculating euclidean distance between image and label map points
Author : Anupama Rajkumar
Date : 27.05.2020
Description: This function is used to calculate the Euclidean distance 
between the image and label points which is used later to identify the
nearest neighbors
*************************************************************************/

double KNN::Euclidean(int imgX, int imgY, int labX, int labY) {
	double distance = 0.0;
	distance = sqrt(pow((imgX - labX), 2) + pow((imgY - labY), 2));
	return distance;
}


/***********************************************************************
Generating a label map 
Author : Anupama Rajkumar
Date : 27.05.2020
Description: Idea is to create a single label map from a list of various 
label classes. This map serves as points of reference when trying to classify
patches
*************************************************************************/

void KNN::generateLabelMap(vector<Mat>& label, vector<string>& labelName, Mat& labelMap) {
	/**********************************
	Oberpfaffenhofen
	0 : Unclassified
	1 : city
	2 : field
	3 : forest
	4 : grassland
	5 : street
	***********************************/
	int rows = label[0].rows;
	int cols = label[0].cols;
	for (int cnt = 0; cnt < NUMOFCLASSES; cnt++) {
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				if (labelMap.at<float>(row, col) == 0) {
					if (label[cnt].at<float>(row, col) > 0.) {
						labelMap.at<float>(row, col) = cnt + 1;		    //class of label
					}
				}			
			}
		}
	}

	//write the contents of label map in a csv, for visualization
	string fileName = "distance_list.csv";		
	utils.WriteToFile(labelMap, fileName);
}

/***********************************************************************
Verifying KNN Classifier with random samples - Overloaded Function
Author : Anupama Rajkumar
Date : 28.05.2020
Description : This function counts the number of classes in k neighborhood
Based on which class has the highest count, appropriate class is returned
*************************************************************************/

int KNN::Classify(vector<pair<double, int>>& distVec, int k) {
	int unClass, city, field, forest, grassland, street;
	vector<pair<int, int>> classType;
	classType.reserve(NUMOFCLASSES+1);
	unClass = city = field = forest = grassland = street = 0;
	for (int i = 0; i < k; i++) {
		switch (distVec[i].second) {
			case 0:
				unClass+=1;
				break;
			case 1:
				city+=1;
				break;
			case 2:
				field+=1;
				break;
			case 3:
				forest+=1;
				break;
			case 4:
				grassland+=1;
				break;
			case 5:
				street+=1;
				break;
			default:
				cout << "Invalid classification";
				break;
		}
	}
	for (int cnt = 0; cnt < (NUMOFCLASSES + 1); cnt++) {
		pair<int, int> classCnt;
		switch (cnt) {
		case 0:
			classCnt.first = unClass;
			classCnt.second = cnt;
			break;
		case 1:
			classCnt.first = city;
			classCnt.second = cnt;
			break;
		case 2:
			classCnt.first = field;
			classCnt.second = cnt;
			break;
		case 3:
			classCnt.first = forest;
			classCnt.second = cnt;
			break;
		case 4:
			classCnt.first = grassland;
			classCnt.second = cnt;
			break;
		case 5:
			classCnt.first = street;
			classCnt.second = cnt;
			break;
		default:
			cout << "Invalid classification";
			break;
		}
		classType.push_back(classCnt);
	}
	//sort in descending order
	sort(classType.begin(), classType.end(), greater());
	int classifier;
	//if all the neighboring points are unclassified
	if (city == 0 && field == 0 && forest == 0 && grassland == 0 && street == 0) {
		classifier = classType[0].second;
	}
	else
	{
		if (classType[0].second == 0) {
			classifier = classType[1].second;
		}
		else {
			classifier = classType[0].second;
		}
	}
	return classifier;
}

/***********************************************************************
Verifying KNN Classifier with random samples - Overloaded Function
Author : Anupama Rajkumar
Date : 28.05.2020
Description : This function counts the number of classes in k neighborhood
Based on which class has the highest count, appropriate class is returned
*************************************************************************/
int KNN::Classify(vector<int> classResult) {
	int unClass, city, field, forest, grassland, street;
	vector<pair<int, int>> classType;
	classType.reserve(NUMOFCLASSES + 1);
	unClass = city = field = forest = grassland = street = 0;
	for (int i = 0; i < classResult.size(); i++) {
		switch (classResult[i]) {
		case 0:
			unClass += 1;
			break;
		case 1:
			city += 1;
			break;
		case 2:
			field += 1;
			break;
		case 3:
			forest += 1;
			break;
		case 4:
			grassland += 1;
			break;
		case 5:
			street += 1;
			break;
		default:
			cout << "Invalid classification";
			break;
		}
	}
	for (int cnt = 0; cnt < (NUMOFCLASSES + 1); cnt++) {
		pair<int, int> classCnt;
		switch (cnt) {
		case 0:
			classCnt.first = unClass;
			classCnt.second = cnt;
			break;
		case 1:
			classCnt.first = city;
			classCnt.second = cnt;
			break;
		case 2:
			classCnt.first = field;
			classCnt.second = cnt;
			break;
		case 3:
			classCnt.first = forest;
			classCnt.second = cnt;
			break;
		case 4:
			classCnt.first = grassland;
			classCnt.second = cnt;
			break;
		case 5:
			classCnt.first = street;
			classCnt.second = cnt;
			break;
		default:
			cout << "Invalid classification";
			break;
		}
		classType.push_back(classCnt);
	}
	//sort in descending order
	sort(classType.begin(), classType.end(), greater());
	int classifier;
	//if all the neighboring points are unclassified, the image point cannot be classified
	if (city == 0 && field == 0 && forest == 0 && grassland == 0 && street == 0) {
		classifier = classType[0].second;
	}
	else
	{
		//if still unclassified, find the next best class
		if (classType[0].second == 0) {
			classifier = classType[1].second;
		}
		else {
			classifier = classType[0].second;
		}
	}
	return classifier;
}



/***********************************************************************
Verifying KNN Classifier with random samples
Author : Anupama Rajkumar
Date : 28.05.2020
Description: This function is similar to KNNTrain, except that it classifies
a group of random points (eg 10) in image and in label map and tries to classify it
Work TBD: Work more on this function and give the user the choice to either 
classify a patch or sigle points
*************************************************************************/

void KNN::KNNTest(vector<Point2i>& samplesImg, vector<Point2i>& samplesLab, Mat& RGBImg, Mat& LabelMap, int k) {
	/*for each sample drawn from the RGB image, calculate it's distance
	 from each label point that has been drawn from the label map*/
	vector<pair<double, int>> distVec;
	for (int imgCnt = 0; imgCnt < samplesImg.size(); imgCnt++) {
		for(int labCnt = 0; labCnt < samplesLab.size(); labCnt++){
			pair<double, int> dist;
			dist.first  = this->Euclidean(samplesImg[imgCnt].x, samplesImg[imgCnt].y, samplesLab[labCnt].x, samplesLab[labCnt].y);
			dist.second = LabelMap.at<float>(samplesLab[labCnt].x, samplesLab[labCnt].y);
			distVec.push_back(dist);			
		}
		sort(distVec.begin(), distVec.end());
		cout << "Point is " << samplesImg[imgCnt].x << "x" << samplesImg[imgCnt].y << ":";
		Classify(distVec, k);
	}
}


/***********************************************************************
Training KNN Classifier with RGB Image
Author : Anupama Rajkumar
Date : 29.05.2020
Description: The idea is to find the nearest neighbor (k) from a given patch
size in the labelMap that corresponds to the point in the RGB Image.
Work TBD : allow user to enter k and patch size. Right now it's hardcoded
*************************************************************************/

void KNN::KNNTrain(Mat& RGBImg, Mat& LabelMap, int k) {
	/*for each pixel in RGB image, depending on distance from it's 
	neighboring pixels, classify image*/
	cout << "Starting training the image file ........." << endl;
	int sizeOfPatch = 50;
	int pStart_r, pStart_c, pEnd_r, pEnd_c;
	pStart_r = pStart_c = pEnd_r = pEnd_c = 0;
	Mat classMap;
	classMap = Mat::zeros(LabelMap.size(), CV_32FC3);
	//image patch start and end hardcoded now - will be made configurable
	//for each row and column in the patch
	for (int row = 1100; row < 1300; row++) {	
		for (int col = 1200; col < LabelMap.cols; col++) {												
			Point2i imgPoint;
			imgPoint.x = row;
			imgPoint.y = col;			
			vector<int> classResult;
			//calculate the start and end of the patch from label map for each point in the image patch
			utils.GetLabelPatchIndex(sizeOfPatch, imgPoint, LabelMap, pStart_r, pStart_c, pEnd_r, pEnd_c);
			for (int i = pStart_r; i < pEnd_r; i++) {
				vector<pair<double, int>> distVec;
				for (int j = pStart_c; j < pEnd_c; j++) {
					pair<double, int> dist;
					//calculate the euclidean distance between each point in the image patch and the label map
					dist.first  = this->Euclidean(row, col, i, j);
					dist.second = LabelMap.at<float>(i, j);
					distVec.push_back(dist);
 				}
				//sort the distance in the ascending order
				sort(distVec.begin(), distVec.end());
				//classify for each row the label patch
				int classVal = this->Classify(distVec, k);
				classResult.push_back(classVal);
			}
			//from a vector of probable classes within k nearest neighbor
			//find the class with highest possibility of occurance
			int finalClass = this->Classify(classResult);
			classMap.at<float>(row, col) = finalClass;
			cout << "Point is " << row << "x" << col << ":";
			utils.DisplayClassName(finalClass);			
		}
	}
	//write the contents of class map in a csv, for visualization
	string fileName = "img_classified.csv";
	utils.WriteToFile(classMap, fileName);	
	cout << "Image training ended!!!" << endl;
}

/***********************************************************************
KNN Classifier
Author : Anupama Rajkumar
Date : 27.05.2020
This function calls the functions to generate the label map and train the 
image patches to classify the points
Work TBD: write a function such that user can either classify a single point
or a patch of RGB image as is happening in KNNTrain
*************************************************************************/
void KNN::KNNClassifier(vector<Mat>& label, vector<string>& labelName, int k, Mat& RGBImg) {
	RGBImg.convertTo(RGBImg, CV_32FC3);
	cv::imwrite("ModifiedRGB.png", RGBImg);
	Mat labelMap = Mat::zeros(RGBImg.size(), CV_32FC3);
	this->generateLabelMap(label, labelName, labelMap);

	//Training the Image 
	this->KNNTrain(RGBImg, labelMap, k);

	this->VisualizationImages(labelMap.size());

	//get n random samples from RGBImg
#if 0
	Data data;
	vector<Point2i> samplesImg;
	int numOfImgSamples;
	numOfImgSamples = 10;
	samplesImg.reserve(numOfImgSamples);
	data.ExtractImagePoints(numOfImgSamples, RGBImg, samplesImg);

	//get n random samples from labelmap
	vector<Point2i> samplesLabel;
	int numOfLabSamples;
	numOfLabSamples = 10;
	samplesLabel.reserve(numOfLabSamples);
	data.ExtractImagePoints(numOfLabSamples, labelMap, samplesLabel);

	//Pass the random samples for classification using KNN
	KNNTest(samplesImg, samplesLabel, RGBImg, labelMap, k);
#endif
}

void KNN::VisualizationImages(Size size) {	

	cout << "Starting visualization..." << endl;
	//visualizing the label map
	string fileName1 = "distance_list.csv";
	string imageName1 = "LabelMap.png";
	utils.Visualization(fileName1, imageName1, size);

	//visualizing the classified map
	string fileName2 = "img_classified.csv";
	string imageName2 = "ClassifiedMap.png";
	utils.Visualization(fileName2, imageName2, size);

	cout << "Visualization complete!!!" << endl;
}
