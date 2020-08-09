
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

	/*	1 -> City
		2 -> Field
		3 -> Forest
		4 -> Grassland
		5 -> Street		*/
	
	/*load all the labels*/
	data.loadLabels(argv[3], data.labelNames, data.labelImages, data.numOfPoints);
	cout << "Labels loaded" << endl;
	/*Check if the coherence matrix features csv for the image is already present. If yes, open and read it*/
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
	/*Check if the coherence matrix features csv for the image is already present. If no, load PolSAR data, 
	calculate the coherence matrix for all the points in the image, store the data in csv to be read next time*/
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







