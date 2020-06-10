#include "Utils.h"

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

/***********************************************************************
A helper function to store the classification data
Author : Anupama Rajkumar
Date : 27.05.2020
Description: Using this function store classification values in csv files
that can be used later for data analysis
*************************************************************************/

void Utils::WriteToFile(Mat& labelMap, string& fileName) {
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