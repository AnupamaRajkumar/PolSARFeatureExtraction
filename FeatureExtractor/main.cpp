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
#include "KNN.h"
#include "Feature.h"
#include "Utils.h"
#include "mp.hpp"
#include "Autoencoder.h"


using namespace std;
using namespace cv;
namespace fs = std::filesystem;



void DivideTrainTestData(int numberOfTrainSamples,
						 int numberOfTestSamples,
						 Data& data, int fold);

void WriteCoherenceMatValues(vector<pair<vector<float>, unsigned char>>& imgData, string& fileName, bool isApp);
void WriteCoherenceMatValues(vector<vector<float>>& featureVector, string& fileName, bool isApp);
void calculateMeanFeatureVector(vector<vector<float>>& featureVector, Mat& outPut);
void ConvertToCoherenceVector(vector<vector<float>>& result, vector<vector<float>>& coherenceVec);

int main(int argc, char** argv)
{
	cout << "In main!!" << endl;
	/*********Variables Area*************/
	int k, kSize;
	int numberOfTrainSamples;
	int numberOfTestSamples;
	bool train;
	//Object of class
	Data data;
	KNN knn;
	Feature feature;
	Utils utils;
	string featureName;
	int choice, numOfSamples, outChoice;
	

	/*********Variable Initialization****************/
	k = 10;
	kSize = 3;
	numberOfTrainSamples = 5;
	numberOfTestSamples = 1;
	numOfSamples = 100;
	choice = 1;
	outChoice = 1;

	ifstream CoherencevecList;
	string fileName = "CoherenceVectorList.csv";
	string outFile = "FeatureVector.csv";
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
		ConvertToCoherenceVector(result, coherenceVec);
		/*create a map of result and label*/

		for (int cnt = 0; cnt < coherenceVec.size(); cnt++) {	
			pair<vector<float>, unsigned char> val;
			 val.first = coherenceVec[cnt];
			 val.second = labelMap[cnt];
			 imgData.push_back(val);
		}
		WriteCoherenceMatValues(imgData, fileName, false);
	}

	/*autoencoder hyperparameters*/
	int inputDim, hiddenDim, epoch;
	double learningRate, momentum;
	inputDim = coherenceVec[0].size();
	hiddenDim = 5;
	learningRate = 0.1;
	momentum = 0.9;
	epoch = 50;
	Autoencoder *aeEncoder = new Autoencoder(inputDim, hiddenDim, learningRate, momentum);

	cout << "---Training menu---" << endl;
	cout << "1. Train entire image" << endl;
	cout << "2. Train patches/samples" << endl;
	cout << "3. Read Feature Vector from the csv" << endl;
	cout << "Please enter your choice (1/2/3) ?" << endl;
	cin >> choice;
	
	if (choice == 1) {
			cout << "Training entire image....." << endl;
			/*pass the points in coherency vector through autoencoder
			Here we're calculating features pixel by pixel*/
			int ctr = 1;
			Mat meanMat;
			cout << "Starting training...." << endl;
			for (int cnt = 0; cnt < coherenceVec.size(); cnt++) {
				aeEncoder->InitializeWts();
				aeEncoder->InitializeBias();
				for (int e = 0; e < epoch; e++) {
					//cout << "Epoch :" << e + 1 << endl;
					aeEncoder->train(coherenceVec[cnt], e, cnt);
				}
				if (cnt % 10000 == 0) {
					cout << cnt << "Samples trained" << endl;
				}
			}
	}
	else if (choice == 2) {
		cout << "Please enter the number of samples to be trained under autoencoder:" << endl;
		cin >> numOfSamples;
		//random samples generator
		std::random_device rd;													 // obtain a random number from hardware
		std::mt19937 eng(rd());													// seed the generator
		std::uniform_int_distribution<> distr(0, coherenceVec.size());		   // define the range
		int start = distr(eng);
		int end = 0;
		int ctr = 0;
		if ((start + numOfSamples) > coherenceVec.size()) {
			end = coherenceVec.size();
		}
		else {
			end = start + numOfSamples;
		}
		for (int cnt = 0; cnt < coherenceVec.size(); cnt++) {
			aeEncoder->InitializeWts();
			aeEncoder->InitializeBias();
			for (int e = 0; e < epoch; e++) {
				//cout << "Epoch :" << e + 1 << endl;
				aeEncoder->train(coherenceVec[cnt], e, ctr);
			}
			ctr++;
			lab.push_back(labelName[cnt]);
			if (cnt % 10000 == 0) {
				cout << cnt << " samples trained" << endl;
			}
		}
	}
	else if (choice == 3) {
		cout << "Reading the feature vector csv..." << endl;
		ifstream featureFile;
		featureFile.open(outFile);
		if (featureFile) {
			cv::Ptr<cv::ml::TrainData> raw_data = cv::ml::TrainData::loadFromCSV(outFile, 0, -2, 0);
			Mat data = raw_data->getSamples();
			for (int row = 0; row < data.rows; row++) {
				vector<float> featureVec;
				for (int col = 0; col < data.cols; col++) {					
					featureVec.push_back(data.at<float>(row, col));
				}
				aeEncoder->m_featureVector.push_back(featureVec);
			}
		}		
	}
	else {
		cerr << "Please enter a valid choice" << endl;
		exit(-1);
	}

	cout << "---Operation menu---" << endl;
	cout << "1. Classify the images" << endl;
	cout << "2. Visualise the feature vector" << endl;
	cout << "3. Write the feature vector to csv" << endl;
	cout << "4. Write the mean matrix to csv" << endl;
	cout << "5. Visualize the classified image" << endl;
	cout << "Please enter your choice (1/2/3/4/5) ?" << endl;
	cin >> outChoice;

	if (outChoice == 1) {
		vector<Mat> trainVal, testVal;
		vector<unsigned char> trainLabel, testLabel;
		cout << "Enter number of training samples per label class" << endl;
		cin >> numberOfTrainSamples;
		//cout << "Enter the number of test samples per label class" << endl;
		//cin >> numberOfTestSamples;

		cout << "Splitting dataset into training and testing.." << endl;
		//Splitting training and testing data for classification
		for (int fold = 0; fold < 5; fold++) {
			cout << "In " << fold + 1 << "......" << endl;
			DivideTrainTestData(numberOfTrainSamples, numberOfTestSamples, data, fold);

			cout << "Number of training samples: " << data.trainSamples.Samples.size() << endl;
			cout << "Number of test samples:" << data.testSamples.Samples.size() << endl;
			int rSize = 6640;
			int colSize = 1390;

			/*****resetting vectors and variables***********/
			int cnt1 = 0, cnt2 = 0;
			trainVal.clear();
			trainLabel.clear();
			testVal.clear();
			testLabel.clear();

			for (const auto& p : data.trainSamples.Samples) {
				unsigned int idx = (p.x * colSize) + p.y;
				//cout << "idx:" << idx;
				vector<float> fVal = aeEncoder->m_featureVector[idx];
				/*convert vector to Mat*/
				Mat fMat;
				fMat = Mat::zeros(1, fVal.size(), CV_32FC1);
				memcpy(fMat.data, fVal.data(), fVal.size() * sizeof(float));
				trainVal.push_back(fMat);
				trainLabel.push_back(data.trainSamples.labelName[cnt1]);
				cnt1++;
			}
			for (const auto& p : data.testSamples.Samples) {
				unsigned int idx = (p.x * colSize) + p.y;
				//cout << "idx:" << idx;
				vector<float> fVal = aeEncoder->m_featureVector[idx];
				/*convert vector to Mat*/
				Mat fMat;
				fMat = Mat::zeros(1, fVal.size(), CV_32FC1);
				memcpy(fMat.data, fVal.data(), fVal.size() * sizeof(float));
				testVal.push_back(fMat);
				testLabel.push_back(data.testSamples.labelName[cnt2]);
				cnt2++;
			}

			knn.KNNTest(trainVal, trainLabel, testVal, testLabel, k);
		}
	}
	else if (outChoice == 2) {
		cout << "Visualising the feature vector...." << endl;
		Mat meanMat;
		int row = 0, col = 0;
		cout << "Enter number of rows and columns needed (row x col):" << endl;
		cin >> row >> col;
		meanMat = Mat(row, col, CV_32FC1);
		calculateMeanFeatureVector(aeEncoder->m_featureVector, meanMat);
		/*Generate colormap*/
		cout << "Generating colormap" << endl;
		Mat outMat;
		meanMat.convertTo(meanMat, CV_8UC1);
		outMat = meanMat.clone();
		//outMat.convertTo(outMat, CV_32FC3);
		//applyCustomColorMap(meanMat, outMat);
		//applyColorMap(meanMat, outMat, COLORMAP_JET);
		//utils.Visualization(meanMat, outMat);
		cv::Mat mlookUpTable_8UC1(1, 256, CV_8UC1);
		for (int i = 0; i < 256; ++i)
		{
			mlookUpTable_8UC1.at<uchar>(0, i) = uchar(255 - i);
		}
		cv::LUT(meanMat, mlookUpTable_8UC1, outMat);
		imwrite("PatchColorMap.png", outMat);	
	}
	else if (outChoice == 3) {
		cout << "Writing feature vector to file.." << endl;
		WriteCoherenceMatValues(aeEncoder->m_featureVector, outFile, false);
	}
	else if (outChoice == 4) {
		/*write to file*/
		cout << "Writing mean feature values per pixel to file.." << endl;
		Mat meanMat;
		int row = 0, col = 0;
		cout << "Enter number of rows and columns needed (row x col):" << endl;
		cin >> row >> col;
		meanMat = Mat(row, col, CV_32FC1);
		calculateMeanFeatureVector(aeEncoder->m_featureVector, meanMat);

		ofstream meanMatPtr;
		string fileName = "meanMat.csv";
		meanMatPtr.open(fileName, ofstream::out);
		for (int row = 0; row < meanMat.rows; row++) {
			for (int col = 0; col < meanMat.cols; col++) {
				meanMatPtr << meanMat.at<float>(row, col) << ",";
			}
			meanMatPtr << endl;
		}		
	}
	else if (outChoice == 5) {
		cout << "Visualising the classified map..." << endl;
	}
	else {
		cerr << "Incorrect input..exiting" << endl;
		exit(-1);
	}

	waitKey(0);
	return 0;	
}



void calculateMeanFeatureVector(vector<vector<float>>& featureVector, Mat& outPut) {
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
		memcpy(OutMat.data, outVec.data(), outVec.size()*sizeof(float));
	}

	/*scale values of outMat to outPut*/
	for (int row = 0; row < OutMat.rows; row++) {
		for (int col = 0; col < OutMat.cols; col++) {
			float val = OutMat.at<float>(row, col) * 255.0;
			outPut.at<float>(row, col) = val;
		}
	}

}

void ConvertToCoherenceVector(vector<vector<float>>& result, vector<vector<float>>& coherenceVec) {
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

void WriteCoherenceMatValues(vector<vector<float>>& featureVector, string& fileName, bool isApp) {
	ofstream coherenceFPtr;
	if (!isApp) {
		coherenceFPtr.open(fileName, ofstream::out);
	}
	else {
		coherenceFPtr.open(fileName, ofstream::out|ofstream::app);
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
void WriteCoherenceMatValues(vector<pair<vector<float>, unsigned char>>& imgData, string& fileName, bool isApp) {
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
void DivideTrainTestData(int numberOfTrainSamples, 
						 int numberOfTestSamples,
						 Data& data, int fold) {
	int offset = int(data.labelImages[0].cols / 5);
	int test_start_idx = int((fold)*offset);
	int test_end_idx = int((fold + 1)*offset);

	cout << test_start_idx << "," << test_end_idx << endl;

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
				//if (testCnt < numberOfTestSamples) {
				data.testSamples.Samples.push_back(newSample);
				data.testSamples.labelName.push_back(cnt + 1);
				testCnt++;
				//}
			}
			else
			{
				/*Ensure that the number of points is less than the max points*/
				if (trainCnt < numberOfTrainSamples) {
					data.trainSamples.Samples.push_back(newSample);
					data.trainSamples.labelName.push_back(cnt + 1);
					trainCnt++;
				}
			}
		}
	}
}







