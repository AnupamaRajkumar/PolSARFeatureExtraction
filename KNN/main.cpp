/*Implementing complete processing chain*
*step 1: Read data - from oberpfaffenhofen - rat, label, image
*step 2: Simple feature extraction
*step 3: Train a classifier (KNN/RF/CNN)
*step 4: Apply trained classifier to test data
*step 5: Visualize - PCA/tSNE? and Evaluate data
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
						 Data& data);


int getSafeSamplePoints(Point2i samplePoint, Data& data, int samplesPerClass, int cnt);

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
	

	/*********Variable Initialization****************/
	k = 1;
	kSize = 3;
	numberOfTrainSamples = 5;
	numberOfTestSamples = 1;
	int choice, numOfTrainingSamples, numOfTestSamples;

	ifstream CoherencevecList;
	string fileName = "CoherenceVectorList.csv";
	string outFile = "FeatureVector.csv";
	vector<vector<float>> coherenceVec;
	vector<pair<vector<float>, unsigned char>> imgData;
	vector<unsigned char> labelName;

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
			for (int col = 0; col < data.cols; col++) {
				if (col == 0) {
					labelName.push_back(data.at<int>(row, col));
				}
				else {
					colData.push_back(data.at<float>(row, col));
				}				
			}
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

	/*autoencoder constructor*/
	int inputDim, hiddenDim, epoch;
	double learningRate, momentum;
	inputDim = coherenceVec[0].size();
	hiddenDim = 5;
	learningRate = 0.1;
	momentum = 1;
	epoch = 50;
	Autoencoder *aeEncoder = new Autoencoder(inputDim, hiddenDim, learningRate, momentum);

	cout << "---Training menu---" << endl;
	cout << "1. Train entire image" << endl;
	cout << "2. Train patches/samples" << endl;
	cout << "3. Read existing data from a csv" << endl;
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
				if ((cnt + 1) % 100000 == 0) {
					cout << cnt + 1 << "Samples trained" << endl;
					WriteCoherenceMatValues(aeEncoder->m_featureVector, outFile, true);
					calculateMeanFeatureVector(aeEncoder->m_featureVector, meanMat);
				}
			}
	}
	else if (choice == 2) {
		cout << "Please enter the number of samples to be trained under autoencoder:" << endl;
		cin >> numOfTrainingSamples;
		//random samples generator
		std::random_device rd;													 // obtain a random number from hardware
		std::mt19937 eng(rd());													// seed the generator
		std::uniform_int_distribution<> distr(0, coherenceVec.size());		   // define the range
		int start = distr(eng);
		int end = 0;
		int ctr = 0;
		vector<unsigned char> lab;
		Mat meanMat;
		int row = 0, col = 0;
		cout << "Enter number of rows and columns needed (row x col):" << endl;
		cin >> row >> col;
		if ((start + numOfTrainingSamples) > coherenceVec.size()) {
			end = coherenceVec.size();
		}
		else {
			end = start + numOfTrainingSamples;
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


		
#if 0
		meanMat = Mat(row, col, CV_32FC1);
		calculateMeanFeatureVector(aeEncoder->m_featureVector, meanMat);

		/*write to file*/
		cout << "Writing to file.." << endl;
		ofstream meanMatPtr;
		string fileName = "meanMat.csv";
		meanMatPtr.open(fileName, ofstream::out);
		for (int row = 0; row < meanMat.rows; row++) {
			for (int col = 0; col < meanMat.cols; col++) {
				meanMatPtr << meanMat.at<float>(row, col) << ",";
			}
			meanMatPtr << endl;
		}
#endif

#if 0
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
#endif
	}
	else if (choice == 3) {
		cout << "do something..to be completed" << endl;
	}
	else {
		cerr << "Please enter a valid choice" << endl;
		exit(-1);
	}

#if 0
	cout << "Enter number of training samples per label class" << endl;
	cin >> numOfTrainingSamples;
	cout << "Enter the number of test samples per label class" << endl;
	cin >> numOfTestSamples;

	cout << "Splitting dataset into training and testing.." << endl;
	//Splitting training and testing data for classification
	DivideTrainTestData(numberOfTrainSamples, numberOfTestSamples, data);

	cout << "Training samples:" << data.trainSamples.Samples.size() << endl;
	cout << "Training Labels:" << data.trainSamples.labelName.size() << endl;
	cout << "Testing samples:" << data.testSamples.Samples.size() << endl;
	cout << "Testing Labels:" << data.testSamples.labelName.size() << endl;
#endif

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
	//OutMat.convertTo(OutMat, CV_8UC1);
	/*scale values of outMat to outPut*/
	for (int row = 0; row < OutMat.rows; row++) {
		for (int col = 0; col < OutMat.cols; col++) {
			float val = OutMat.at<float>(row, col) * 255.0;
			outPut.at<float>(row, col) = val;
			//cout << outPut.at<float>(row, col) << " ";
		}
		//cout << endl;
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
						 Data& data) {
	int training_start_idx = int(data.labelImages[0].cols / 5);

	int trainCnt, testCnt;
	//random samples generator
	std::random_device rd;										   // obtain a random number from hardware
	std::mt19937 eng(rd());										   // seed the generator
	std::uniform_int_distribution<> distrX(0, data.labelImages[0].rows);		   // define the range
	std::uniform_int_distribution<> distrY(0, data.labelImages[0].cols);

	/*The idea is to get a balanced division between all the classes. 
	5 classes with equal number of points. Also, the first 1/5th region is 
	reserved for testing data set and from remaining area training samples are taken*/
	int samplesPerClass = int(numberOfTrainSamples / NUMOFCLASSES);
	/*for each class*/
	for (int cnt = 0; cnt < data.numOfPoints.size(); cnt++) {										
		trainCnt = 0;
		testCnt = 0;
		/*for each point in each class*/
		for (int pt = 0; pt < data.numOfPoints[cnt].size(); pt++) {
			int x = distrX(eng);
			int y = distrY(eng);
			Point2i newSample(data.numOfPoints[cnt][pt].x, data.numOfPoints[cnt][pt].y);
			if (newSample.y > training_start_idx) {
				//cout << "newsample:" << newSample.x << "x" << newSample.y << endl;
				//cout << pt << trainCnt << endl;
				/*Ensure that the number of points is less than the max points*/
				if (trainCnt < samplesPerClass) {
					int val = getSafeSamplePoints(newSample, data, samplesPerClass, cnt);
					if (val == 1) {						
						data.trainSamples.Samples.push_back(newSample);
						data.trainSamples.labelName.push_back(cnt+1);		//data.labelNames[cnt]				
						trainCnt++;
					}
				}
			}
			else
			{
				if (testCnt < numberOfTestSamples) {
					int val = getSafeSamplePoints(newSample, data, samplesPerClass, cnt);
					if (val == 1) {
						data.testSamples.Samples.push_back(newSample);
						data.testSamples.labelName.push_back(cnt+1);			//data.labelNames[cnt]
						testCnt++;
					}
				}				
			}
		}
	}
}


int getSafeSamplePoints(Point2i samplePoint, Data& data, int samplesPerClass, int cnt) {

	Point2i new_ind;
	int j_min = samplePoint.x - int(data.sizeOfPatch / 2); 
	int j_max = samplePoint.x + int(data.sizeOfPatch / 2);
	int i_min = samplePoint.y - int(data.sizeOfPatch / 2);
	int i_max = samplePoint.y + int(data.sizeOfPatch / 2);
	// get rid of the points on the borders
	if (i_max < data.labelImages[cnt].cols && j_max < data.labelImages[cnt].rows && i_min >= 0 && j_min >= 0) {
		// get rid of points which are half patch size away from the mask zero area
			return 1;				
	}
	else {
		return 0;
	}
}







