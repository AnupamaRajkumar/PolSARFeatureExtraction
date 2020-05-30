#include "KNN.h"
#include "Data.h"

#include <iostream>
#include <math.h>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

double Euclidean(int imgX, int imgY, int labX, int labY) {
	double distance = 0.0;
	distance = sqrt(pow((imgX - labX), 2) + pow((imgY - labY), 2));
	return distance;
}


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

void Visualization(Mat& labelMap) {
	Mat dispLabelMap = labelMap.clone();
	dispLabelMap.rows = labelMap.rows;
	dispLabelMap.cols = labelMap.cols;
	int val;
	//cout << labelMap.rows << "x" << labelMap.cols << endl;
	for (int row = 0; row < dispLabelMap.rows; row++) {
		for (int col = 0; col < dispLabelMap.cols; col++) {						//labelMap.cols
			val = dispLabelMap.at<float>(row, col);								//labelMap.at<float>(row, col);
			//cout << "val:" << val << endl;
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
	Visualization(labelMap);
}

/***********************************************************************
Verifying KNN Classifier with random samples
Author : Anupama Rajkumar
Date : 28.05.2020
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
	pair<int, int> classifier;
	//if all the neighboring points are unclassified
	if (city == 0 && field == 0 && forest == 0 && grassland == 0 && street == 0) {
		classifier = classType[0];
	}
	else
	{
		if (classType[0].second == 0) {
			classifier = classType[1];
		}
		else {
			classifier = classType[0];
		}
	}
	

	switch (classifier.second) {
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
			cout << classifier.second << endl;
			cout << "Something went wrong..can't be classified" << endl;
			break;
	}
	return classifier.second;
}


/***********************************************************************
Verifying KNN Classifier with random samples
Author : Anupama Rajkumar
Date : 28.05.2020
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
Training KNN Classifier with RGB Image
Author : Anupama Rajkumar
Date : 29.05.2020
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
	for (int row = 5656; row < 6500; row++) {													
		for (int col = 950; col < 1300; col++) {												
			Point2i imgPoint;
			imgPoint.x = row;
			imgPoint.y = col;			
			vector<pair<double, int>> distPatch;
			GetLabelPatchIndex(sizeOfPatch, imgPoint, LabelMap, pStart_r, pStart_c, pEnd_r, pEnd_c);
			//cout << pStart_r << "," << pStart_c << "," << pEnd_r << "," << pEnd_c << endl;
			for (int i = pStart_r; i < pEnd_r; i++) {
				vector<pair<double, int>> distVec;
				for (int j = pStart_c; j < pEnd_c; j++) {
					pair<double, int> dist;
					dist.first  = Euclidean(row, col, i, j);
					dist.second = LabelMap.at<float>(i, j);
					distVec.push_back(dist);
				}
				sort(distVec.begin(), distVec.end());
				distPatch.push_back(distVec[0]);
			}
			sort(distPatch.begin(), distPatch.end());
			//sort(distVec.begin(), distVec.end());
			cout << "Point is " << row << "x" << col << ":";
			int classVal = Classify(distPatch, k);
			classMap.at<float>(row, col) = classVal;
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
