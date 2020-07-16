/*Implementing complete processing chain*
*step 1: Read data - from oberpfaffenhofen - rat, label, image -
*step 2: Simple feature extraction - 
*step 3: Train a classifier (KNN/RF/CNN) - 
*step 4: Apply trained classifier to test data -
*step 5: Visualize - 
*/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <algorithm>
#include <fstream>
#include <map>

#include "Data.h"
#include "Feature.h"
#include "mp.hpp"
#include "Autoencoder.h"
#include "Utils.h"


using namespace std;
using namespace cv;
namespace fs = std::filesystem;


int main(int argc, char** argv)
{
	cout << "In main!!" << endl;
	/*********Variables Area*************/

	//Object of class
	Data data;
	Feature feature;
	Utils utils;
	Autoencoder ae;

	/*********Variable Initialization****************/
	ifstream CoherencevecList;
	string fileName = "CoherenceVectorList.csv";
	vector<vector<float>> coherenceVec;
	vector<pair<vector<float>, unsigned char>> imgData;
	vector<unsigned char> labelName;
	vector<unsigned char> lab;

	/*********Function calls****************/
	//load PolSAR data
	//data.loadData(argv[1]);
	//cout << "Data loaded" << endl;

	/*	1 -> City
		2 -> Field
		3 -> Forest
		4 -> Grassland
		5 -> Street		*/
	
	data.loadLabels(argv[3], data.labelImages, data.labelNames, data.numOfPoints);
	cout << "Labels loaded" << endl;
	CoherencevecList.open(fileName);
	if (CoherencevecList) {
		/*read the contents from file*/
		//reading data from csv
		cv::Ptr<cv::ml::TrainData> raw_data = cv::ml::TrainData::loadFromCSV(fileName, 0, -2, 0);
		Mat data = raw_data->getSamples();
		for (int row = 0; row < data.rows; row++) {
			vector<float> colData;	
			int name;
			for (int col = 0; col < data.cols; col++) {
				if (col == 0) {
					name = data.at<int>(row, 0);
				}
				else {
					colData.push_back(data.at<float>(row, col));
				}				
			}
			labelName.push_back(name);
			coherenceVec.push_back(colData);
		}
	}
	else {
		//load PolSAR data
		data.loadData(argv[1]);
		cout << "Data loaded" << endl;
		cout << "Calculating coherency matrix" << endl;
		vector<vector<float>> result;
		vector<unsigned char> labelMap;
		feature.GetCoherencyFeatures(data, result, labelMap);
		utils.ConvertToCoherenceVector(result, coherenceVec);
		/*create a map of result and label*/

		for (int cnt = 0; cnt < coherenceVec.size(); cnt++) {	
			pair<vector<float>, unsigned char> val;
			 val.first = coherenceVec[cnt];
			 val.second = labelMap[cnt];
			 imgData.push_back(val);
		}
		utils.WriteCoherenceMatValues(imgData, fileName, false);
	}

	/*Autoencoder User Options*/
	ae.AutoencoderUserMenu(coherenceVec, data);

	waitKey(0);
	return 0;	
}







