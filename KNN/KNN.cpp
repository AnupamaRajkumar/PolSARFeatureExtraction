#include "KNN.h"
#include "Data.h"

#include <iostream>
#include <math.h>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/***********************************************************************
Calculationg euclidean distance between image and label map points
Author : Anupama Rajkumar
Date : 27.05.2020
Description: This function is used to calculate the Euclidean distance 
between the image and label points which is used later to identify the
nearest neighbors
*************************************************************************/

double Euclidean(int imgX, int imgY, int labX, int labY) {
	double distance = 0.0;
	distance = sqrt(pow((imgX - labX), 2) + pow((imgY - labY), 2));
	return distance;
}

/***********************************************************************
A helper function to store the classification data
Author : Anupama Rajkumar
Date : 27.05.2020
Description: Using this function store classification values in csv files
that can be used later for data analysis
*************************************************************************/

void WriteToFile(Mat& labelMap, string& fileName) {
	ofstream distance_list;
	distance_list.open(fileName);								

	for (int row = 0; row < labelMap.rows; row++) {
		for (int col = 0; col < labelMap.cols; col++) {
			distance_list << labelMap.at<float>(row, col) << ",";
		}
		distance_list << endl;
	}
}

/***********************************************************************
A helper function to visualize the maps (label or classified)
Author : Anupama Rajkumar
Date : 27.05.2020
Description: Using this function to create visualizations by reading the stored
csv files
Work TBD : Incomplete function. Still working
*************************************************************************/
void Visualization(Mat& labelMap) {
	Mat dispLabelMap = labelMap.clone();
	dispLabelMap.rows = labelMap.rows;
	dispLabelMap.cols = labelMap.cols;
	int val;
	for (int row = 0; row < dispLabelMap.rows; row++) {
		for (int col = 0; col < dispLabelMap.cols; col++) {						//labelMap.cols
			val = dispLabelMap.at<float>(row, col);								//labelMap.at<float>(row, col);
			switch (val) {
			case 0:
				dispLabelMap.at<float>(row, col) = 0.;						    //unclassified - black
				break;
			case 1:
				dispLabelMap.at<float>(row, col) = 10.;							//Vec3i(0, 0, 128) : city - brown
				break;
			case 2:
				dispLabelMap.at<float>(row, col) = 55.;							//Vec3i(0, 128, 255) : field - Orange
				break;
			case 3:
				dispLabelMap.at<float>(row, col) = 125.;						//Vec3i(0, 255, 0) forest - green
				break;
			case 4:
				dispLabelMap.at<float>(row, col) = 180.;						//Vec3i(255, 0, 255) grassland - magenta
				break;
			case 5:
				dispLabelMap.at<float>(row, col) = 200.;						//Vec3i(0, 255, 255) street - yellow
				break;
			default:
				cout << "Wrong value" << endl;
				break;
			}
			//cout << "col:" << col << endl;
		}
		//cout << "row:" << row << endl;
	}
	cv::imwrite("LabelMap.png", dispLabelMap);
}

/***********************************************************************
Generating a label map 
Author : Anupama Rajkumar
Date : 27.05.2020
Description: Idea is to create a single label map from a list of various 
label classes. This map serves as points of reference when trying to classify
patches
*************************************************************************/

void generateLabelMap(vector<Mat>& label, vector<string>& labelName, Mat& labelMap) {
	/**********************************
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
				if (label[cnt].at<float>(row, col) > 0.) {
					labelMap.at<float>(row, col) = cnt + 1;		    //class of label
				}
				else if (labelMap.at<float>(row, col) != 0) {
					//do nothing
				}
				else {
					labelMap.at<float>(row, col) = 0;
				}					
			}
		}
	}

	//write the contents of label map in a csv, for visualization
	string fileName = "distance_list.csv";
	WriteToFile(labelMap, fileName);

	//visualizing the label map
	//Visualization(labelMap);
}

/***********************************************************************
Verifying KNN Classifier with random samples - Overloaded Function
Author : Anupama Rajkumar
Date : 28.05.2020
Description : This function counts the number of classes in k neighborhood
Based on which class has the highest count, appropriate class is returned
*************************************************************************/

int Classify(vector<pair<double, int>>& distVec, int k) {
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
int Classify(vector<int> classResult) {
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

void KNNTest(vector<Point2i>& samplesImg, vector<Point2i>& samplesLab, Mat& RGBImg, Mat& LabelMap, int k) {
	/*for each sample drawn from the RGB image, calculate it's distance
	 from each label point that has been drawn from the label map*/
	vector<pair<double, int>> distVec;
	for (int imgCnt = 0; imgCnt < samplesImg.size(); imgCnt++) {
		for(int labCnt = 0; labCnt < samplesLab.size(); labCnt++){
			pair<double, int> dist;
			dist.first  = Euclidean(samplesImg[imgCnt].x, samplesImg[imgCnt].y, samplesLab[labCnt].x, samplesLab[labCnt].y);
			dist.second = LabelMap.at<float>(samplesLab[labCnt].x, samplesLab[labCnt].y);
			distVec.push_back(dist);			
		}
		sort(distVec.begin(), distVec.end());
		cout << "Point is " << samplesImg[imgCnt].x << "x" << samplesImg[imgCnt].y << ":";
		Classify(distVec, k);
	}
}

/***********************************************************************
Helper function to get the patch start and end index from label map
Author : Anupama Rajkumar
Date : 29.05.2020
Description: This function provides the start and end positions of the patch
from which the nearest neighbour needs to be found in the labelMap
*************************************************************************/

void GetLabelPatchIndex(int sizeOfPatch, Point2i samplePoints, Mat& LabelMap, int& pStart_r, int& pStart_c, int& pEnd_r, int& pEnd_c) {
	pStart_r = samplePoints.x - sizeOfPatch / 2;
	pStart_c = samplePoints.y - sizeOfPatch / 2;

	pEnd_r = samplePoints.x + sizeOfPatch / 2;
	pEnd_c = samplePoints.y + sizeOfPatch / 2;

	if (pStart_r < 0 || pStart_c < 0 )
	{
		pStart_r = samplePoints.x;
		pStart_c = samplePoints.y;
		pEnd_r = samplePoints.x + sizeOfPatch;
		pEnd_c = samplePoints.y + sizeOfPatch;
	}
	if (pEnd_r > LabelMap.rows || pEnd_c > LabelMap.cols) {
		pStart_r = samplePoints.x - sizeOfPatch;
		pStart_c = samplePoints.y - sizeOfPatch;
		pEnd_r	 = samplePoints.x;
		pEnd_c   = samplePoints.y;
	}
}

/***********************************************************************
Helper function to print the name of the identified class
Author : Anupama Rajkumar
Date : 29.05.2020
Description: This function prints the  name of the class identified
*************************************************************************/
void DisplayClassName(int finalClass) {
	switch (finalClass) {
	case 1:
		cout << "Point classified as City" << endl;
		break;
	case 2:
		cout << "Point classified as Field" << endl;
		break;
	case 3:
		cout << "Point classified as Forest" << endl;
		break;
	case 4:
		cout << "Point classified as Grassland" << endl;
		break;
	case 5:
		cout << "Point classified as Street" << endl;
		break;
	default:
		cout << finalClass << endl;
		cout << "Something went wrong..can't be classified" << endl;
		break;	}

}

/***********************************************************************
Training KNN Classifier with RGB Image
Author : Anupama Rajkumar
Date : 29.05.2020
Description: The idea is to find the nearest neighbor (k) from a given patch
size in the labelMap that corresponds to the point in the RGB Image.
Work TBD : allow user to enter k and patch size. Right now it's hardcoded
*************************************************************************/

void KNNTrain(Mat& RGBImg, Mat& LabelMap, int k) {
	/*for each pixel in RGB image, depending on distance from it's 
	neighboring pixels, classify image*/
	cout << "Starting training the image file ........." << endl;
	int sizeOfPatch = 30;
	int pStart_r, pStart_c, pEnd_r, pEnd_c;
	pStart_r = pStart_c = pEnd_r = pEnd_c = 0;
	Mat classMap;
	classMap = Mat::zeros(LabelMap.size(), CV_32FC1);
	//image patch start and end hardcoded now - will be made configurable
	//for each row and column in the patch
	for (int row = 5656; row < 6500; row++) {													
		for (int col = 950; col < 1300; col++) {												
			Point2i imgPoint;
			imgPoint.x = row;
			imgPoint.y = col;			
			vector<int> classResult;
			//calculate the start and end of the patch from label map for each point in the image patch
			GetLabelPatchIndex(sizeOfPatch, imgPoint, LabelMap, pStart_r, pStart_c, pEnd_r, pEnd_c);
			for (int i = pStart_r; i < pEnd_r; i++) {
				vector<pair<double, int>> distVec;
				for (int j = pStart_c; j < pEnd_c; j++) {
					pair<double, int> dist;
					//calculate the euclidean distance between each point in the image patch and the label map
					dist.first  = Euclidean(row, col, i, j);
					dist.second = LabelMap.at<float>(i, j);
					distVec.push_back(dist);
				}
				//sort the distance in the ascending order
				sort(distVec.begin(), distVec.end());
				//classify for each row the label patch
				int classVal = Classify(distVec, k);
				classResult.push_back(classVal);
			}
			//from a vector of probable classes within k nearest neighbor
			//find the class with highest possibility of occurance
			int finalClass = Classify(classResult);
			classMap.at<float>(row, col) = finalClass;
			cout << "Point is " << row << "x" << col << ":";
			DisplayClassName(finalClass);			
		}
	}
	//write the contents of class map in a csv, for visualization
	string fileName = "img_classified.csv";
	WriteToFile(classMap, fileName);	
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
	RGBImg.convertTo(RGBImg, CV_32FC1);
	cv::imwrite("ModifiedRGB.png", RGBImg);
	Mat labelMap = Mat::zeros(RGBImg.size(), CV_32FC1);
	generateLabelMap(label, labelName, labelMap);

	//Training the Image 
	KNNTrain(RGBImg, labelMap, k);

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
