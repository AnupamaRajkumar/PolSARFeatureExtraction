#include "Utils.h"
#include "Data.h"


#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <random>

using namespace std;
using namespace cv;

/***********************************************************************
A helper function to store the classification data
Author : Anupama Rajkumar
Date : 27.05.2020
Modified by Anupama on 16.06.2020
Description: Using this function store classification values in csv files
that can be used later for data analysis
*************************************************************************/

void Utils::WriteToFile(int k, double accuracy, int trainSize, int testSize, string& featureName) {

	fstream performance_log;
	string fileName = "PerformanceVal.csv";
	cout << "feature name:" << featureName << endl;
	performance_log.open(fileName, fstream::in|fstream::out|fstream::app);
	performance_log << featureName << "," << k << "," << trainSize << "," << testSize << " " << accuracy <<  endl;
	performance_log.close();
}




/***********************************************************************
A helper function containing color metadata to be used when visualizing
Author : Eli Ionescu
Date : 27.05.2020
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
			int val = image.at<float>(row, col);
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


/***********************************************************************
A helper function to visualize the maps (label or classified)
Author : Anupama Rajkumar
Date : 01.06.2020
Description: Using this function to create visualizations by reading the stored
csv files
*************************************************************************/

void Utils::Visualization(string& fileName, string& imageName, Size size) {

	cv::Mat img;
	//reading data from csv
	cv::Ptr<cv::ml::TrainData> raw_data = cv::ml::TrainData::loadFromCSV(fileName, 0, -1, -1);
	cv::Mat data = raw_data->getSamples();
	// optional if you have a color image and not just raw data
	data.convertTo(img, CV_32FC1);
	// set the image size
	cv::resize(img, img, size);
	//visualize the map
	visualiseLabels(img, imageName);
	cv::waitKey(0);
}



/***********************************************************************
Helper function to get the patch start and end index from label map
Author : Anupama Rajkumar
Date : 29.05.2020
Description: This function provides the start and end positions of the patch
from which the nearest neighbour needs to be found in the labelMap
*************************************************************************/

void Utils::GetLabelPatchIndex(int sizeOfPatch, Point2i samplePoints, Mat& LabelMap, int& pStart_r, int& pStart_c, int& pEnd_r, int& pEnd_c) {
	/*Ensure that the patch size is even*/
	if ((sizeOfPatch % 2) != 0) {
		cout << "Please ensure that the patch size is even. Changing patch dimension to next lower even number" << endl;
		sizeOfPatch = -1;
	}
	int rowStart, rowEnd, colStart, colEnd;
	pStart_r = samplePoints.x - (sizeOfPatch / 2.);
	pStart_c = samplePoints.y - (sizeOfPatch / 2.);

	pEnd_r = samplePoints.x + (sizeOfPatch / 2.);
	pEnd_c = samplePoints.y + (sizeOfPatch / 2.);

	if ((pStart_r < 0) || (pStart_c < 0))
	{
		pStart_r = 0;
		pStart_c = 0;
		pEnd_r = sizeOfPatch;
		pEnd_c = sizeOfPatch;
	}
	if ((pEnd_r > LabelMap.rows) || (pEnd_c > LabelMap.cols))
	{
		pEnd_r = LabelMap.rows - 1;
		pEnd_c = LabelMap.cols - 1;
		pStart_r = LabelMap.rows - 1 - sizeOfPatch;
		pStart_c = LabelMap.cols - 1 - sizeOfPatch;
	}
}

/***********************************************************************
Helper function to print the name of the identified class
Author : Anupama Rajkumar
Date : 29.05.2020
Description: This function prints the  name of the class identified
*************************************************************************/
void Utils::DisplayClassName(int finalClass) {
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
		break;
	}

}


/***********************************************************************
Generating a individual label map
Author : Anupama Rajkumar
Date : 27.05.2020
Description: Generating csv and visualisation of individual labels
*************************************************************************/


void Utils::generateTestLabel(vector<Mat>& label, vector<string>& labelName, Mat& labelMap, int cnt) {
	/**********************************
Oberpfaffenhofen
0 : Unclassified
1 : city
2 : field
3 : forest
4 : grassland
5 : street
***********************************/
	cout << labelName[cnt] << endl;
	for (int row = 0; row < label[cnt].rows; row++) {
		for (int col = 0; col < label[cnt].cols; col++) {
			if (labelMap.at<float>(row, col) == 0.0f) {
				if (label[cnt].at<float>(row, col) > 0.0f) {
					labelMap.at<float>(row, col) = cnt + 1;		    //class of label
				}
			}
		}
	}
	string fileName, imageName;
	switch (cnt) {
	case 0:
		fileName = "city.csv";
		//this->WriteToFile(labelMap, fileName);		
		imageName = "city.png";
		this->Visualization(fileName, imageName, labelMap.size());
		break;
	case 1:
		fileName = "field.csv";
		//this->WriteToFile(labelMap, fileName);
		imageName = "field.png";
		this->Visualization(fileName, imageName, labelMap.size());
		break;
	case 2:
		fileName = "forest.csv";
		//this->WriteToFile(labelMap, fileName);
		imageName = "forest.png";
		this->Visualization(fileName, imageName, labelMap.size());
		break;
	case 3:
		fileName = "grassland.csv";
		//this->WriteToFile(labelMap, fileName);
		imageName = "grassland.png";
		this->Visualization(fileName, imageName, labelMap.size());
		break;
	case 4:
		fileName = "streets.csv";
		//this->WriteToFile(labelMap, fileName);
		imageName = "streets.png";
		this->Visualization(fileName, imageName, labelMap.size());
		break;
	default:
		break;
	}
}

/***********************************************************************
Generating a label map
Author : Anupama Rajkumar
Date : 27.05.2020
Description: Idea is to create a single label map from a list of various
label classes. This map serves as points of reference when trying to classify
patches
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
	//this->WriteToFile(labelMap, fileName);
}

/***********************************************************************
Visualising the images 
Author : Anupama Rajkumar
Date : 27.05.2020
Description: This function generates the visualisation with the csv as 
input
*************************************************************************/

void Utils::VisualizationImages(Size size) {

	cout << "Starting visualization..." << endl;
	//visualizing the label map
	string fileName1 = "distance_list.csv";
	string imageName1 = "LabelMap.png";
	this->Visualization(fileName1, imageName1, size);

	//visualizing the classified map
	string fileName2 = "img_classified.csv";
	string imageName2 = "ClassifiedMap.png";
	this->Visualization(fileName1, imageName2, size);

	cout << "Visualization complete!!!" << endl;
}


void Utils::calculateMeanFeatureVector(vector<vector<float>>& featureVector, Mat& outPut) {
	vector<float> outVec;
	for (int cnt = 0; cnt < featureVector.size(); cnt++) {
		float mean = 0.;
		for (int s = 0; s < featureVector[cnt].size(); s++) {
			mean += featureVector[cnt][s];
			mean /= featureVector[cnt].size();
		}
		outVec.push_back(mean);
	}
	/*convert vector to Mat*/
	Mat OutMat;
	OutMat = Mat::zeros(outPut.rows, outPut.cols, CV_32FC1);
	if (outVec.size() == outPut.rows * outPut.cols) {
		memcpy(OutMat.data, outVec.data(), outVec.size() * sizeof(float));
	}

	/*scale values of outMat to outPut*/
	for (int row = 0; row < OutMat.rows; row++) {
		for (int col = 0; col < OutMat.cols; col++) {
			float val = OutMat.at<float>(row, col) * 255.0;
			outPut.at<float>(row, col) = val;
		}
	}

}

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

/*override*/
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


/************************************************************
Dividing the data samples into training and test samples
Take some training samples for each class and same for
test samples
Date: 11.06.2020
Author: Anupama Rajkumar
*************************************************************/
void Utils::DivideTrainTestData(int numberOfTrainSamples, int numberOfTestSamples, Data& data, int fold) {
	int offset = int(data.labelImages[0].cols / 5);
	int test_start_idx = int((fold)*offset);
	int test_end_idx = int((fold + 1)*offset);

	cout << test_start_idx << "," << test_end_idx << endl;
	numberOfTestSamples = (int)(0.2 * numberOfTrainSamples);

	int trainCnt, testCnt;
	//random samples generator
	std::random_device rd;															// obtain a random number from hardware
	std::mt19937 eng(rd());															// seed the generator
	std::uniform_int_distribution<> distrX(0, data.labelImages[0].rows);			// define the range
	std::uniform_int_distribution<> distrY(0, data.labelImages[0].cols);
	cout << numberOfTrainSamples << " " << numberOfTestSamples << endl;
	/*The idea is to get a balanced division between all the classes.
	5 classes with equal number of points. Also, the first 1/5th region is
	reserved for testing data set and from remaining area training samples are taken*/
	//int samplesPerClass = int(numberOfTrainSamples / NUMOFCLASSES);
	/*for each class*/
	data.trainSamples.Samples.clear();
	data.trainSamples.labelName.clear();
	data.testSamples.Samples.clear();
	data.testSamples.labelName.clear();

	for (int cnt = 0; cnt < data.numOfPoints.size(); cnt++) {
		trainCnt = 0;
		testCnt = 0;
		/*for each point in each class*/
		for (int pt = 0; pt < data.numOfPoints[cnt].size(); pt++) {
			int x = distrX(eng);
			int y = distrY(eng);
			Point2i newSample(data.numOfPoints[cnt][pt].x, data.numOfPoints[cnt][pt].y);
			//cout << "newsample:" << newSample.x << "x" << newSample.y << endl;
			//cout << pt << trainCnt << endl;
			if ((newSample.y > test_start_idx) && (newSample.y < test_end_idx)) {
				if (testCnt < numberOfTestSamples) {
					data.testSamples.Samples.push_back(newSample);
					data.testSamples.labelName.push_back(cnt + 1);
					testCnt++;
				}
				if (trainCnt < numberOfTrainSamples) {
					data.trainSamples.Samples.push_back(newSample);
					data.trainSamples.labelName.push_back(cnt + 1);
					trainCnt++;
				}
			}
			/*else
			{
				//Ensure that the number of points is less than the max points
				if (trainCnt < numberOfTrainSamples) {
					data.trainSamples.Samples.push_back(newSample);
					data.trainSamples.labelName.push_back(cnt + 1);
					trainCnt++;
				}
			}*/
		}
	}
}