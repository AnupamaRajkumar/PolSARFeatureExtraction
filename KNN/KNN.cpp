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
	distance = sqrt(pow((imgX - labX), 2) + pow((imgY, labY), 2));
	return distance;
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
	int cols = label[1].cols;
	for (int cnt = 0; cnt < NUMOFCLASSES; cnt++) {
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				if (label[cnt].at<float>(row, col) == 255) {
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
	ofstream distance_list;
	distance_list.open("distance_list.csv");

	for (int row = 0; row < labelMap.rows; row++) {
		for (int col = 0; col < labelMap.cols; col++) {
			distance_list << labelMap.at<float>(row, col) << ",";
		}
		distance_list << endl;
	}

	cout << labelMap.channels() << labelMap.channels() << endl;
	//visualizing the label map
	//Mat dispLabelMap = Mat::zeros(labelMap.size(), CV_32FC1);
	/*Mat dispLabelMap = labelMap.clone();  
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
					dispLabelMap.at<float>(row, col) = 128.;					    //Vec3i(0, 0, 128) : city - brown
					break;
				case 2:
					dispLabelMap.at<float>(row, col) = 33023.;						//Vec3i(0, 128, 255) : field - Orange
					break;
				case 3:
					dispLabelMap.at<float>(row, col) = 65280.;						//Vec3i(0, 255, 0) forest - green
					break;
				case 4:
					dispLabelMap.at<float>(row, col) = 16711935.;						//Vec3i(255, 0, 255) grassland - magenta
					break;
				case 5:
					dispLabelMap.at<float>(row, col) = 65535.;						//Vec3i(0, 255, 255) street - yellow
					break;
				default:
					cout << "Wrong value" << endl;
					break;
			}
			//cout << "col:" << col << endl;
		}
		//cout << "row:" << row << endl;
	}
	cv::imwrite("LabelMap.png", dispLabelMap);*/
}


void Classify(vector<pair<double, int>>& distVec, int k) {
	int unClass, city, field, forest, grassland, street;
	//int classType[NUMOFCLASSES+1];
	vector<pair<int, int>> classType;
	classType.reserve(NUMOFCLASSES+1);
	//int n = sizeof(classType) / sizeof(classType[0]);
	unClass = city = field = forest = grassland = street = 0;
	for (int i = 0; i < k; i++) {
		pair<int, int> classVal;
		int classCnt;
		switch (distVec[i].second) {
			case 0:
				classCnt = ++unClass;
				break;
			case 1:
				classCnt = ++city;
				break;
			case 2:
				classCnt = ++field;
				break;
			case 3:
				classCnt = ++forest;
				break;
			case 4:
				classCnt = ++grassland;
				break;
			case 5:
				classCnt = ++street;
				break;
			default:
				cout << "Invalid classification";
				classCnt = -1;
				break;
		}
		classVal.first = classCnt;
		classVal.second = distVec[i].second;
		classType.push_back(classVal);
	}

	sort(classType.begin(), classType.end());
	pair<int, int> classifier;
	
	if (classType[NUMOFCLASSES].second == 0) {
		classifier = classType[NUMOFCLASSES-1];
	}
	else {
		classifier = classType[NUMOFCLASSES];
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
			//cout << "Label value at " << samplesLab[labCnt].x << "x" << samplesLab[labCnt].y << " is:" << LabelMap.at<float>(samplesLab[labCnt].x, samplesLab[labCnt].y)<< endl;
			//cout << labCnt << ":" << dist.first << "\t" << dist.second << endl;
			//cout << "distance of " << samplesImg[imgCnt].x << "," << samplesImg[imgCnt].y << "from" << samplesLab[labCnt].x << "," << samplesLab[labCnt].y << "is:" << dist << endl;
			distVec.push_back(dist);
			sort(distVec.begin(), distVec.end());
		}
		/*for (int cnt = 0; cnt < distVec.size(); cnt++) {
			cout << distVec[cnt].first << "\t" << distVec[cnt].second << endl;
		}*/
		cout << "Point is " << samplesImg[imgCnt].x << "x" << samplesImg[imgCnt].y << ":";
		Classify(distVec, k);
	}
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
	//KNNTrain();
	//get n random samples from RGBImg
	Data data;
	vector<Point2i> samplesImg;
	int numOfImgSamples;
	numOfImgSamples = 10;
	samplesImg.reserve(numOfImgSamples);
	data.ExtractImagePoints(numOfImgSamples, RGBImg, samplesImg);
	/*for (int cnt = 0; cnt < numOfImgSamples; cnt++) {
		cout << samplesImg[cnt].x << " " << samplesImg[cnt].y << endl;
	}*/
	//get n random samples from labelmap
	vector<Point2i> samplesLabel;
	int numOfLabSamples;
	numOfLabSamples = 10;
	samplesLabel.reserve(numOfLabSamples);
	data.ExtractImagePoints(numOfLabSamples, labelMap, samplesLabel);
	/*for (int cnt = 0; cnt < numOfLabSamples; cnt++) {
		cout << samplesLabel[cnt].x << " " << samplesLabel[cnt].y << endl;
	}*/

	//Pass the random samples for classification using KNN
	//KNNTest(samplesImg, samplesLabel, RGBImg, labelMap, k);
}
