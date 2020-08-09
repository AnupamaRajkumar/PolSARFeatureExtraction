#include "Utils.h"
#include "Data.h"


#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <random>

using namespace std;
using namespace cv;



/***********************************************************************
A helper function containing color metadata to be used when visualizing
Author : Eli Ionescu
Description: Using this function creates a map of the colors and the labels
they correspond to. To be used with visualization
*************************************************************************/
map<string, Vec3f> Utils::loadLabelsMetadata()
{
	map<string, Vec3f> name_color;

	// Color is BGR not RGB!
	Vec3f red = Vec3f(49.0f, 60.0f, 224.0f);
	Vec3f blue = Vec3f(164.0f, 85.0f, 50.0f);
	Vec3f yellow = Vec3f(0.0f, 190.0f, 246.0f);
	Vec3f dark_green = Vec3f(66.0f, 121.0f, 79.0f);
	Vec3f light_green = Vec3f(0.0f, 189.0f, 181.0f);
	Vec3f black = Vec3f(0.0f, 0.0f, 0.0f);

	name_color["city"] = red;
	name_color["field"] = yellow;
	name_color["forest"] = dark_green;
	name_color["grassland"] = light_green;
	name_color["street"] = blue;
	name_color["unclassified"] = black;

	return name_color;
}

/***********************************************************************
A helper function to visualize the maps (label or classified)
Author : Eli Ionescu
Date : 27.05.2020
Description: Using this function to assign colors to maps (label and classified)
Input 1 : Image matrix
Input 2 : Name of the generated image
Output  : Generated image stored as png
*************************************************************************/
Mat_<Vec3f> Utils::visualiseLabels(Mat &image, string& imageName)
{
	map<string, Vec3f> colors = loadLabelsMetadata();

	Mat result = Mat(image.rows, image.cols, CV_32FC3, Scalar(255.0f, 255.0f, 255.0f));
	// Create the output result;
	for (int row = 0; row < image.rows; row++) {
		for (int col = 0; col < image.cols; col++) {
			Vec3f color;
			// Take every point and assign the right color in the result Mat
			int val = image.at<int>(row, col);
			switch (val) {
			case 0:
				color = colors["unclassified"];
				break;
			case 1:
				color = colors["city"];
				break;
			case 2:
				color = colors["field"];
				break;
			case 3:
				color = colors["forest"];
				break;
			case 4:
				color = colors["grassland"];
				break;
			case 5:
				color = colors["street"];
				break;
			default:
				cout << "Wrong value" << endl;
				break;
			}
			result.at<Vec3f>(row, col) = color;
		}
	}
	imwrite(imageName, result);

	return result;
}


/***************************************************************************
Generating a label map
Author : Anupama Rajkumar
Description: This function creates a single label map from a list of various
label classes. This map serves as points of reference when trying to classify
patches
Input 1 : label images
Output  : generated label map stored in a distance csv
*************************************************************************/

void Utils::generateLabelMap(vector<Mat>& label, Mat& labelMap) {
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
		;
		for (int row = 0; row < label[cnt].rows; row++) {
			for (int col = 0; col < label[cnt].cols; col++) {
				if (labelMap.at<unsigned char>(row, col) == 0) {
					if (label[cnt].at<float>(row, col) > 0.0f) {
						labelMap.at<unsigned char>(row, col) = unsigned char(cnt + 1);		    //class of label
					}
				}
			}
		}
	}

	//write the contents of label map in a csv, for visualization
	string fileName = "distance_list.csv";
}

/***************************************************************************
Helper function to generate a coherence vector for each pixel
Author : Anupama Rajkumar
Description: This function generates a 9x1 coherence vector for each pixel
Input 1 : the coherence vector values of entire image which is a vector of 9
dimension and each vector has (6640x1390) values
Output  : coherency matrix of each pixel for entire image. 9x1 vector for
entire image consisting of 6640x1390 pixels
****************************************************************************/

void Utils::ConvertToCoherenceVector(vector<vector<float>>& result, vector<vector<float>>& coherenceVec) {
	unsigned int maxLen = result[0].size();
	for (int len = 0; len < maxLen; len++) {
		vector<float> cohVec;
		for (int cnt = 0; cnt < result.size(); cnt++) {
			float val = result[cnt].at(len);
			cohVec.push_back(val);
		}
		coherenceVec.push_back(cohVec);
	}
}


/****************************************************************************************
A helper function to store the classification data
Author : Anupama Rajkumar
Description: Use this function to store calculated values in csv files
that can be used later for data analysis
Input 1 : input vector
Input 2 : name of the csv file
Input 3 : is data to be appended to existing file?
****************************************************************************************/

void Utils::WriteCoherenceMatValues(vector<vector<float>>& featureVector, string& fileName, bool isApp) {
	ofstream coherenceFPtr;
	if (!isApp) {
		coherenceFPtr.open(fileName, ofstream::out);
	}
	else {
		coherenceFPtr.open(fileName, ofstream::out | ofstream::app);
	}

	for (int cnt = 0; cnt < featureVector.size(); cnt++) {
		for (int len = 0; len < featureVector[cnt].size(); len++) {
			coherenceFPtr << featureVector[cnt].at(len) << ",";
		}
		coherenceFPtr << endl;
	}
	coherenceFPtr.close();
}


/************************************************************************************
A helper function to store the classification data - overloaded function
Author : Anupama Rajkumar
Description: Use this function to store calculated values in csv files
that can be used later for data analysis
Input 1 : input vector and label pair
Input 2 : name of the csv file
Input 3 : is data to be appended to existing file?
**************************************************************************************/
void Utils::WriteCoherenceMatValues(vector<pair<vector<float>, unsigned char>>& imgData, string& fileName, bool isApp) {
	ofstream coherenceFPtr;
	if (!isApp) {
		coherenceFPtr.open(fileName, ofstream::out);
	}
	else {
		coherenceFPtr.open(fileName, ofstream::out | ofstream::app);
	}

	for (int cnt = 0; cnt < imgData.size(); cnt++) {
		coherenceFPtr << (int)imgData[cnt].second << ",";
		for (int len = 0; len < imgData[cnt].first.size(); len++) {
			coherenceFPtr << imgData[cnt].first.at(len) << ",";
		}
		coherenceFPtr << endl;
	}
	coherenceFPtr.close();

}

/******************************************************************************************
Test/Train data split : This function splits data into test and training data patches
Author : Anupama Rajkumar
Description: This function splits entire image into 5 patches and stores the data in these
patches which will be used as training and testing sets
Input 1  : Patch index
Input 2  : Input image data points for all classes
Output 1 : Point and label pair for each patch for the entire image
*******************************************************************************************/
vector<pair<vector<Point2i>, uint>> Utils::GetPatchPoints(int patchIdx, Data& data) {
	int offset = int(data.labelImages[0].cols / MAXNUMOFPATCHES);
	int start_idx = int((patchIdx)*offset);
	int end_idx = int((patchIdx + 1)*offset);

	vector<pair<vector<Point2i>, uint>> patchPoint;

	for (int cnt = 0; cnt < data.numOfPoints.size(); cnt++) {
		/*for each point in each class*/
		//cout << "cnt :" << cnt << endl;
		vector<Point2i> point;
		pair<vector<Point2i>, uint> pPt;
		for (int len = 0; len < data.numOfPoints[cnt].size(); len++) {
			/*if the point lies within the patch*/			
			Point2i newPoint(data.numOfPoints[cnt][len].x, data.numOfPoints[cnt][len].y);
			if ((newPoint.y > start_idx) && (newPoint.y <= end_idx)) {	
				point.push_back(newPoint);
			}
		}
		pPt.first = point;
		pPt.second = cnt + 1;
		patchPoint.push_back(pPt);
		cout << cnt + 1 << ": "<< patchPoint[cnt].first.size() << endl;
	}
	return patchPoint;
}



/**************************************************************************
Dividing the data samples into training and test samples
Author: Anupama Rajkumar
Description: This function uses the split data patches and uses 20% points as
test samples and 80% points as training samples. The test samples is shuffled
depending on the number of the fold and cross validation is achieved. The number
of test and training samples are chosen such that the class points are balanced.
Input 1 : Inout image data
Input 2 : Fold number
Input 3 : Patch index
***************************************************************************/
void Utils::DivideTrainTestData(Data& data, int fold, int patchIdx) {
	/*calculate offset based on the total number of patches - 5 in this case*/
	int patchOffset = int(data.labelImages[0].cols / MAXNUMOFPATCHES);
	/*depending on the current patch index, determine patch start and end index*/
	int pStartIdx = int((patchIdx)*patchOffset);
	int pEndIdx = int((patchIdx + 1)*patchOffset);

	int trainCnt, testCnt;
	int start_idx = 0;
	int end_idx = 0;
	int offset = 0; 

	offset = int(patchOffset / 5);
	start_idx = int(fold * offset) + pStartIdx;
	end_idx = int((fold + 1)*offset) + pStartIdx;

	/*for each class*/
	data.trainSamples.Samples.clear();
	data.trainSamples.labelName.clear();
	data.testSamples.Samples.clear();
	data.testSamples.labelName.clear();

	cout << pStartIdx << "," << pEndIdx << "," << start_idx << ","  << end_idx << endl;

	for (int cnt1 = 0; cnt1 < data.numOfPoints.size(); cnt1++) {
		/*for each point in each class*/
		for (int len = 0; len < data.numOfPoints[cnt1].size(); len++) {
			/*if the point lies within the patch*/
			Point2i newPoint(data.numOfPoints[cnt1][len].x, data.numOfPoints[cnt1][len].y);
			/*if points within the current patch, use as training sample else use as training sample*/
			if ((newPoint.y >= pStartIdx) && (newPoint.y < pEndIdx)) {
				if ((newPoint.y >= start_idx) && (newPoint.y < end_idx)) {
					data.testSamples.Samples.push_back(newPoint);
					data.testSamples.labelName.push_back(cnt1+1);
				}
				else {
					data.trainSamples.Samples.push_back(newPoint);
					data.trainSamples.labelName.push_back(cnt1+1);
				}					
			}
		}
	}
}

/**************************************************************************
Dividing the data samples into training and test samples - overloaded
Author: Anupama Rajkumar
Description: This function uses the split data patches and uses 20% points as
test samples and 80% points as training samples. The test samples is shuffled
depending on the number of the fold and cross validation is achieved. The number
of test and training samples are chosen such that the class points are balanced.
Input 1 : Inout image data
Input 2 : Fold number
Input 3 : Patch point determined while splitting data
***************************************************************************/
void Utils::DivideTrainTestData(Data& data, int fold, vector<pair<vector<Point2i>, uint>> patchPoint) {
	int trainCnt, testCnt;
	int start_idx = 0;
	int end_idx = 0;
	int offset = 0;

	/*for each class*/
	data.trainSamples.Samples.clear();
	data.trainSamples.labelName.clear();
	data.testSamples.Samples.clear();
	data.testSamples.labelName.clear();

	/*4/5th in training set, 1/5th in test set*/
	for (int cnt1 = 0; cnt1 < patchPoint.size(); cnt1++) {
		/*for each point in each class*/
		for (int len = 0; len < patchPoint[cnt1].first.size(); len++) {
			offset = patchPoint[cnt1].first.size() / 5;
			/*shuffle the data based on fold number thus ensuring cross-validation*/
			start_idx = int(fold*offset);
			end_idx = int((fold + 1)*offset);
			//cout << start_idx << "," << end_idx << endl;
			Point2i newPoint(patchPoint[cnt1].first[len].x, patchPoint[cnt1].first[len].y);
			/*if within the start and end index (20%) data, use as test data else use as test data*/
			if ((len >= start_idx) && (len < end_idx)) {
				data.testSamples.Samples.push_back(newPoint);
				data.testSamples.labelName.push_back(patchPoint[cnt1].second);
			}
			else {
				data.trainSamples.Samples.push_back(newPoint);
				data.trainSamples.labelName.push_back(patchPoint[cnt1].second);
			}
		}
	}
}

