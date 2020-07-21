#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <fstream>
#include <numeric>

#include "Autoencoder.h"
//#include "StackedAutoencoder.h"
#include "KNN.h"
#include "Utils.h"
#include "Performance.h"


using namespace std;
using namespace cv;
namespace fs = std::filesystem;

void CalculateClassification(bool isAE, Data& data, Utils& utils, int k,
	Autoencoder& aeEncoder, KNN knn, Performance perform);

void Autoencoder::AutoencoderUserMenu(vector<vector<float>>& coherenceVec, Data& data) {

	int choice, numOfSamples, outChoice;
	int numberOfTrainSamples, numberOfTestSamples;
	int k;


	KNN knn;
	Utils utils;
	Performance perform;


	choice = 1;
	numOfSamples = 100;
	outChoice = 1;
	numberOfTrainSamples = 5;
	numberOfTestSamples = 1;
	string outFile = "FeatureVector.csv";
	string stackedFile = "StackedFeatureVector.csv";
	k = 1;

	/*vanilla autoencoder hyperparameters*/
	int inputDim, hiddenDim, hiddenDim1, hiddenDim2, epoch, epoch2;
	double learningRate1, learningRate2, momentum;
	inputDim = coherenceVec[0].size();
	hiddenDim = 5;
	hiddenDim1 = 5;
	hiddenDim2 = 3;
	learningRate1 = 0.1;
	learningRate2 = 0.5;
	momentum = 0.9;
	epoch = 100;
	epoch2 = 100;
	Autoencoder *aeEncoder = new Autoencoder(inputDim, hiddenDim, learningRate1, momentum);
	Autoencoder *aeEncoder2 = new Autoencoder(hiddenDim, hiddenDim2, learningRate2, momentum);
	//StackedAE *stackedAE = new StackedAE(inputDim, hiddenDim1, hiddenDim2, learningRate1, momentum);


	cout << "---Autoencoder Training Menu---" << endl;
	cout << "1. Train entire image using Vanilla AE" << endl;
	cout << "2. Train patches/samples using Vanilla AE" << endl;
	cout << "3. Read Feature Vector from the csv for Vanilla AE" << endl;
	cout << "4. Train entire image using Stacked AE" << endl;
	cout << "5. Read Feature Vector from the csv for Stacked AE" << endl;
	cout << "Please enter your choice (1/2/3/4/5) ?" << endl;
	cin >> choice;

	if (choice == 1) {
		cout << "Training entire image....." << endl;
		/*pass the points in coherency vector through autoencoder
		Here we're calculating features pixel by pixel*/
		cout << "Starting training...." << endl;
		for (int cnt = 0; cnt < coherenceVec.size(); cnt++) {
			aeEncoder->InitializeWts();
			aeEncoder->InitializeBias();
			for (int e = 0; e < epoch; e++) {
				//cout << "Epoch :" << e + 1 << endl;
				aeEncoder->train(coherenceVec[cnt], cnt, e);
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
				aeEncoder->train(coherenceVec[cnt], cnt, e);
			}
			ctr++;
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
	else if (choice == 4) {
		vector<vector<float>> encoderWt, outData;
		vector<float> outputBias, reconsOutput;
		cout << "Training entire image....." << endl;
		/*pass the points in coherency vector through autoencoder
		Here we're calculating features pixel by pixel*/
		Mat meanMat;
		cout << "Starting training...." << endl;		
		for (int cnt = 0; cnt < coherenceVec.size(); cnt++) {
			//cout << "Training AE layer 1 .." << endl;
			aeEncoder->InitializeWts();
			aeEncoder->InitializeBias();
			for (int e = 0; e < epoch; e++) {
				//cout << "Epoch :" << e + 1 << endl;
				aeEncoder->train(coherenceVec[cnt], cnt, e);
			}
			aeEncoder->SaveParameters(encoderWt, outputBias);
			//cout << "Training AE layer 2..." << endl;
			aeEncoder2->InitializeWts();
			aeEncoder2->InitializeBias();
			for (int e = 0; e < epoch2; e++) {
				//cout << "Epoch :" << e + 1 << endl;
				aeEncoder2->train(aeEncoder->m_featureVector[cnt], cnt, e);
			}
			aeEncoder->ReconstructOutput(encoderWt, outputBias, aeEncoder2->m_outputVector[0], reconsOutput);
			outData.push_back(reconsOutput);
			if (cnt % 10000 == 0) {
				cout << cnt << "Samples trained" << endl;
			}
		}
	}
	else if (choice == 5) {
		cout << "Reading the feature vector csv..." << endl;
		ifstream featureFile;
		featureFile.open(stackedFile);
		if (featureFile) {
			cv::Ptr<cv::ml::TrainData> raw_data = cv::ml::TrainData::loadFromCSV(stackedFile, 0, -2, 0);
			Mat data = raw_data->getSamples();
			for (int row = 0; row < data.rows; row++) {
				vector<float> featureVec;
				for (int col = 0; col < data.cols; col++) {
					featureVec.push_back(data.at<float>(row, col));
				}
				aeEncoder2->m_featureVector.push_back(featureVec);
			}
		}
	}
	else {
		cerr << "Please enter a valid choice" << endl;
		exit(-1);
	}

	cout << "---Operation menu---" << endl;
	cout << "1. Classify the AE images" << endl;
	cout << "2. Visualise the AE feature vector" << endl;
	cout << "3. Write the AE feature vector to csv" << endl;
	cout << "4. Write the stacked AE feature vector to csv" << endl;
	cout << "5. Write the AE mean matrix to csv" << endl; 
	cout << "6. Visualise the stacked AE feature vector" << endl;
	cout << "7. Visualize the classified image" << endl;
	cout << "8. Classify the stacked AE images" << endl;
	cout << "9. Visualise the classified image" << endl;
	cout << "Please enter your choice (1/2/3/4/5/6/7/8/9) ?" << endl;
	cin >> outChoice;

	if (outChoice == 1) {
		bool isAE;
		isAE = true;
		CalculateClassification(isAE, data, utils, k, *aeEncoder, knn, perform);
		string perfFile = "Performance_Metrics_AE.csv";
	}
	else if (outChoice == 2) {
		cout << "Visualising the feature vector...." << endl;
		Mat meanMat;
		int row = 0, col = 0;
		cout << "Enter number of rows and columns needed (row x col):" << endl;
		cin >> row >> col;
		meanMat = Mat(row, col, CV_32FC1);
		utils.calculateMeanFeatureVector(aeEncoder->m_featureVector, meanMat);
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
		cv::equalizeHist(outMat, outMat);
		cv::imwrite("PatchColorMap_AE.png", outMat);
	}
	else if (outChoice == 3) {
		cout << "Writing feature vector to file.." << endl;
		utils.WriteCoherenceMatValues(aeEncoder->m_featureVector, outFile, false);
	}
	else if (outChoice == 4) {
		cout << "Writing stacked AE feature vector to file..." << endl;
		utils.WriteCoherenceMatValues(aeEncoder2->m_featureVector, stackedFile, false);
	}
	else if (outChoice == 5) {
		/*write to file*/
		cout << "Writing Autoencoder mean feature values per pixel to file.." << endl;
		Mat meanMat;
		int row = 0, col = 0;
		cout << "Enter number of rows and columns needed (row x col):" << endl;
		cin >> row >> col;
		meanMat = Mat(row, col, CV_32FC1);
		utils.calculateMeanFeatureVector(aeEncoder->m_featureVector, meanMat);

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
	else if (outChoice == 6) {
		cout << "Visualising the stacked AE feature vector...." << endl;
		Mat meanMat;
		int row = 0, col = 0;
		cout << "Enter number of rows and columns needed (row x col):" << endl;
		cin >> row >> col;
		meanMat = Mat(row, col, CV_32FC1);
		utils.calculateMeanFeatureVector(aeEncoder2->m_featureVector, meanMat);
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
		cv::equalizeHist(outMat, outMat);
		cv::imwrite("PatchColorMap_StackedAE.png", outMat);
	}
	else if (outChoice == 7) {
		cout << "Visualising the classified map..." << endl;
	}
	else if (outChoice == 8) {
		bool isAE;
		isAE = false;
		CalculateClassification(isAE, data, utils, k, *aeEncoder2, knn, perform);
	}
	else if (outChoice == 9) {
		string visualizeFile = "InterMediate.csv";
		string imageName = "ClassifiedAE_full_1.png";
		cout << "Visualise the classified image" << endl;
		utils.Visualization(visualizeFile, imageName, data.labelImages[0].size());
	}
	else {
		cerr << "Incorrect input..exiting" << endl;
		exit(-1);
	}
}

void CalculateClassification(bool isAE, Data& data, Utils& utils, int k, 
							 Autoencoder& aeEncoder, KNN knn, Performance perform) {
	string perfFile;
	if (!isAE) {
		perfFile = "Performance_Metrics_SAE.csv";
	}
	else {
		perfFile = "Performance_Metrics_AE.csv";
	}
	
	Mat img;
	Mat classifiedImage;

	int rSize = 6640;
	int colSize = 1390;
	img = Mat::zeros(rSize, colSize, CV_32FC1);
	classifiedImage = Mat::zeros(rSize, colSize, CV_32FC1);

	ofstream perfPtr;
	perfPtr.open(perfFile, ofstream::out | ofstream::app);
	perfPtr << "PatchIdx" << "," << "OA" << endl;
	perfPtr.close();

	for (int patchIdx = 136; patchIdx < MAXNUMOFPATCHES; patchIdx++) {

		vector<pair<vector<Point2i>, uint>> patchPoint = utils.GetPatchPoints(patchIdx, data);
		vector<float> accuracy;
		string fileName, csvFile, classify;
		if (!isAE) {
			fileName = to_string(patchIdx) + "_SAE.png";
			csvFile = to_string(patchIdx) + "_SAE.csv";
			classify = to_string(patchIdx) + "_SAE_classify.csv";
		}
		else {
			fileName = to_string(patchIdx) + "_AE.png";
			csvFile = to_string(patchIdx) + "_AE.csv";
			classify = to_string(patchIdx) + "_AE_classify.csv";
		}
		accuracy.clear();
		cout << "Splitting dataset into training and testing.." << endl;
		//Splitting training and testing data for classification
		for (int fold = 0; fold < 5; fold++) {
			vector<Mat> trainVal, testVal;
			vector<unsigned char> trainLabel, testLabel;
			vector<unsigned char> classResult;

			cout << "In " << fold + 1 << "......" << endl;
			//utils.DivideTrainTestData(data, fold, patchIdx);
			utils.DivideTrainTestData(data, fold, patchPoint);

			cout << "Number of training samples: " << data.trainSamples.Samples.size() << endl;
			cout << "Number of test samples:" << data.testSamples.Samples.size() << endl;

			/*****resetting vectors and variables***********/
			int cnt1 = 0, cnt2 = 0;
			for (const auto& p : data.trainSamples.Samples) {
				unsigned int idx = (p.x * colSize) + p.y;
				//cout << "idx:" << idx;
				vector<float> fVal = aeEncoder.m_featureVector[idx];
				/*convert vector to Mat*/
				Mat fMat;
				fMat = Mat::zeros(1, fVal.size(), CV_32FC1);
				memcpy(fMat.data, fVal.data(), fVal.size() * sizeof(float));
				trainVal.push_back(fMat);
				trainLabel.push_back(data.trainSamples.labelName[cnt1]);
				cnt1++;
				fMat.release();
			}
			for (const auto& p : data.testSamples.Samples) {
				unsigned int idx = (p.x * colSize) + p.y;
				//cout << "idx:" << idx;
				vector<float> fVal = aeEncoder.m_featureVector[idx];
				/*convert vector to Mat*/
				Mat fMat;
				fMat = Mat::zeros(1, fVal.size(), CV_32FC1);
				memcpy(fMat.data, fVal.data(), fVal.size() * sizeof(float));
				testVal.push_back(fMat);
				testLabel.push_back(data.testSamples.labelName[cnt2]);
				cnt2++;
				fMat.release();
			}
			cout << " starting classification.." << endl;
			knn.KNNTest(trainVal, trainLabel, testVal, testLabel, classResult, k);

			float intAccuracy = perform.calculatePredictionAccuracy(classResult, testLabel);
			accuracy.push_back(intAccuracy);
			cout << "Overall Accuracy for fold " << fold + 1 << " : " << intAccuracy << endl;

			for (int cnt = 0; cnt < data.testSamples.Samples.size(); cnt++) {
				Point2i newPoint(data.testSamples.Samples[cnt]);
				classifiedImage.at<int>(newPoint.x, newPoint.y) = (int)classResult[cnt];
			}

			for (int cnt = 0; cnt < trainVal.size(); cnt++) {
				trainVal[cnt].release();
			}
			for (int cnt = 0; cnt < testVal.size(); cnt++) {
				testVal[cnt].release();
			}
			trainLabel.clear();
			testLabel.clear();
			classResult.clear();
		}
		float avgAccuracy = 0.;
		float sumOfElements = 0.;
		for (auto& a : accuracy) {
			sumOfElements += a;
		}
		avgAccuracy = sumOfElements / 5;
		cout << "Sum :" << sumOfElements << endl;
		cout << "Average accuracy of patch " << patchIdx + 1 << ":" << avgAccuracy << endl;

		perfPtr.open(perfFile, ofstream::out | ofstream::app);
		perfPtr << patchIdx + 1 << "," << avgAccuracy << endl;
		perfPtr.close();

		ofstream fileptr;
		fileptr.open(csvFile, ofstream::out);
		for (int row = 0; row < classifiedImage.rows; row++) {
			for (int col = 0; col < classifiedImage.cols; col++) {
				fileptr << classifiedImage.at<int>(row, col) << ",";
			}
			fileptr << endl;
		}

		//if (patchIdx % 5 == 0) {
		utils.visualiseLabels(classifiedImage, fileName);
		//}
	}
}