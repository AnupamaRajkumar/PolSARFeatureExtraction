#include <iostream>
#include <opencv2/opencv.hpp>
#include <random>
#include <fstream>
#include <numeric>

#include "Autoencoder.h"
#include "Visualisation.h"
//#include "KNN.h"
//#include "Utils.h"
//#include "Performance.h"
//#include "Data.h"


using namespace std;
using namespace cv;
namespace fs = std::filesystem;



/*********************************************************************************************
Description: User menu for various operations related to autoencoder
Author : Anupama Rajkumar
Input 1   : Coherence vector calculated for entire image
Input 2   : Image data
**********************************************************************************************/

void Autoencoder::AutoencoderUserMenu(vector<vector<float>>& coherenceVec, Data& data) {

	int choice, numOfSamples, outChoice;
	int numberOfTrainSamples, numberOfTestSamples;
	int k;

	choice = 1;
	numOfSamples = 100;
	outChoice = 1;
	numberOfTrainSamples = 5;
	numberOfTestSamples = 1;
	string outFile = "FeatureVector.csv";
	string stackedFile = "StackedFeatureVector.csv";
	string recostructedFile = "ReconstructedImage.csv";
	k = 1;

	KNN knn;
	Utils utils;
	Performance perform;

	/*vanilla autoencoder hyperparameters*/
	int inputDim, hiddenDim, epoch;
	double learningRate1, momentum;
	inputDim = coherenceVec[0].size();
	hiddenDim = 5;
	learningRate1 = 0.1;
	momentum = 0.9;
	epoch = 100;

	/*Initialise the autoencoder object*/
	Autoencoder *aeEncoder = new Autoencoder(inputDim, hiddenDim, learningRate1, momentum);

	/*Autoencoder training menu*/
	cout << "---Autoencoder Training Menu---" << endl;
	cout << "1. Train entire image using Vanilla AE" << endl;
	cout << "2. Train patches/samples using Vanilla AE" << endl;
	cout << "3. Read Feature Vector from the csv for Vanilla AE" << endl;
	cout << "Please enter your choice (1/2/3) ?" << endl;
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
		cout << "Please enter the number of samples (per class) to be trained under autoencoder:" << endl;
		cin >> numOfSamples;
		cout << "Reading samples...." << endl;
		vector<pair<Point2i, uint>> point;
		int cnt1;
		for (int cnt = 0; cnt < NUMOFCLASSES; cnt++) {
			pair<Point2i, uint> newP;
			int cnt1 = 0;
			for (int len = 0; len < data.numOfPoints[cnt].size(); len++) {
				Point2i newPoint(data.numOfPoints[cnt][len].x, data.numOfPoints[cnt][len].y);
				if (cnt1 < numOfSamples) {
					newP.first = newPoint;
					newP.second = cnt + 1;
					point.push_back(newP);
					cnt1++;
				}
			}
		}
		cout << "Starting training...." << endl;
		for (int cnt = 0; cnt < point.size(); cnt++) {
			aeEncoder->InitializeWts();
			aeEncoder->InitializeBias();
			unsigned int idx = (point[cnt].first.x * 1390) + point[cnt].first.y;
			//cout << idx << endl;
			for (int e = 0; e < epoch; e++) {
				//cout << "Epoch :" << e + 1 << endl;
				aeEncoder->train(coherenceVec.at(idx), cnt, e);
			}
			if (cnt % 1000 == 0) {
				cout << cnt << "Samples trained" << endl;
			}
		}

		ofstream coherenceFPtr;
		string fileName = "AEReducedDim.txt";
		coherenceFPtr.open(fileName, ofstream::out);

		for (int cnt = 0; cnt < point.size(); cnt++) {
			coherenceFPtr << point[cnt].second << ",";
			for (int len = 0; len < aeEncoder->m_featureVector[cnt].size(); len++) {
				coherenceFPtr << aeEncoder->m_featureVector[cnt].at(len) << ",";
			}
			coherenceFPtr << endl;
		}
		coherenceFPtr.close();
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
	cout << "1. Classify the AE images" << endl;
	cout << "2. Visualise the AE feature vector" << endl;
	cout << "3. Write the AE feature vector and output value to csv" << endl;
	cout << "4. Reconstruct the autoencoder output" << endl;
	cout << "5. Calculate differnce between input and reconstructed matrices" << endl;
	cout << "6. Exit" << endl;
	cout << "Please enter your choice (1/2/3/4/5/6) ?" << endl;
	cin >> outChoice;

	if (outChoice == 1) {
		bool isAE;
		isAE = true;
		this->CalculateClassification(isAE, data, utils, k, *aeEncoder, knn, perform);
		string perfFile = "Performance_Metrics_AE.csv";
	}
	else if (outChoice == 2) {
		cout << "Visualising the AE feature vector...." << endl;
		Visual visual;
		string imagePrefix = "PatchColorMapAE_color_";
		/*Read the feature file - csv or hdf5*/
		string outFile = "FeatureVector.csv";
		vector<vector<float>> m_featureVector;
		/*Reading the feature vector file for visualisation*/
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
				m_featureVector.push_back(featureVec);
			}
		}
		visual.GenerateFeatureMap(m_featureVector, imagePrefix);
	}
	else if (outChoice == 3) {
		cout << "Writing feature vector to file.." << endl;
		utils.WriteCoherenceMatValues(aeEncoder->m_featureVector, outFile, false);
		cout << "Writing Reconstructed output vector to file.." << endl;
		utils.WriteCoherenceMatValues(aeEncoder->m_outputVector, recostructedFile, false);
	}
	else if (outChoice == 4) {
		int row = 6640, col = 1390;
		Mat alpha, beta, gamma, result;
		alpha = Mat::zeros(row, col, CV_32FC1);
		beta = Mat::zeros(row, col, CV_32FC1);
		gamma = Mat::zeros(row, col, CV_32FC1);
		vector<float> alVec, beVec, gaVec;
		vector<Mat> reconstData;

		cv::Ptr<cv::ml::TrainData> raw_data = cv::ml::TrainData::loadFromCSV(recostructedFile, 0, -2, 0);		
		Mat recData = raw_data->getSamples();

		for (int row = 0; row < recData.rows; row++) {
			float alVal, beVal, gaVal;
			alVal = 2.0 * sqrt(exp(recData.at<float>(row, 0)));
			beVal = 2.0 * sqrt(exp(recData.at<float>(row, 1)));
			gaVal = sqrt(exp(recData.at<float>(row, 2))) / 2.0;
			alVec.push_back(alVal);
			beVec.push_back(beVal);
			gaVec.push_back(gaVal);
		}
		if (alVec.size() == row * col) {
			memcpy(alpha.data, alVec.data(), alVec.size() * sizeof(float));
			memcpy(beta.data, beVec.data(), beVec.size() * sizeof(float));
			memcpy(gamma.data, gaVec.data(), gaVec.size() * sizeof(float));
		}
		threshold(beta, beta, 2.5 * mean(beta).val[0], 0, THRESH_TRUNC);		//red
		threshold(gamma, gamma, 2.5 * mean(gamma).val[0], 0, THRESH_TRUNC);		//green
		threshold(alpha, alpha, 2.5 * mean(alpha).val[0], 0, THRESH_TRUNC);		//blue

		double minVal, maxVal, maxValR, maxValG, maxValB;
		minMaxLoc(beta, &minVal, &maxValR);
		minMaxLoc(gamma, &minVal, &maxValG);
		minMaxLoc(alpha, &minVal, &maxValB);
		maxVal = max(max(maxValR, maxValG), maxValB);

		for (int rows = 0; rows < row; rows++) {
			for (int cols = 0; cols < col; cols++) {
				alpha.at<float>(rows, cols) *= (255 / maxVal);
				beta.at<float>(rows, cols) *= (255 / maxVal);
				gamma.at<float>(rows, cols) *= (255 / maxVal);
			}
		}

		reconstData.push_back(beta);
		reconstData.push_back(gamma);
		reconstData.push_back(alpha);

		merge(reconstData, result);
		result.convertTo(result, CV_8UC3);
		imwrite("Reconstructed.png", result);

	}
	else if (outChoice == 5) {
		int row = 6640, col = 1390;
		Mat coherencyInMat, coherencyReconsMat, logInMatrix, logReconsMatrix, normMat;
		string fileName = "CoherenceVectorList.csv";
		vector<float> normVec;
		cv::Ptr<cv::ml::TrainData> raw_data = cv::ml::TrainData::loadFromCSV(recostructedFile, 0, -2, 0);		
		Mat recData = raw_data->getSamples();
		cv::Ptr<cv::ml::TrainData> in_data = cv::ml::TrainData::loadFromCSV(fileName, 0, -2, 0);
		Mat inputData = in_data->getSamples();
		coherencyInMat = Mat::zeros(3, 3, CV_32FC1);
		coherencyReconsMat = Mat::zeros(3, 3, CV_32FC1);
		logInMatrix = Mat::zeros(3, 3, CV_32FC1);
		logReconsMatrix = Mat::zeros(3, 3, CV_32FC1);
		normMat = Mat::zeros(row, col, CV_32FC1);
		for (int row = 0; row < recData.rows; row++) {
			/*input data*/
			this->CalculateCoherencyMatrix(inputData, row, coherencyInMat);
			this->CalculateLogMatrix(coherencyInMat, logInMatrix);
			/*reconstructed data*/
			this->CalculateCoherencyMatrix(recData, row, coherencyReconsMat);
			this->CalculateLogMatrix(coherencyReconsMat, logReconsMatrix);
			/*calculate difference and Frobenius Norm*/
			float forbNorm = this->CalculateDifferenceMatrix(logInMatrix, logReconsMatrix);
			/*write norm to a normVector*/
			normVec.push_back(forbNorm);

			if (row % 10000 == 0) {
				cout << row + 1 << " points covered.." << endl;
			}
		}
		if (normVec.size() == row * col) {
			memcpy(normMat.data, normVec.data(), normVec.size() * sizeof(float));
		}
		threshold(normMat, normMat, 2.5 * mean(normMat).val[0], 0, THRESH_TRUNC);
		double minVal, maxVal;
		minMaxLoc(normMat, &minVal, &maxVal);
		for (int rows = 0; rows < row; rows++) {
			for (int cols = 0; cols < col; cols++) {
				normMat.at<float>(rows, cols) *= (255 / maxVal);
			}
		}
		normMat.convertTo(normMat, CV_8UC1);
		equalizeHist(normMat, normMat);
		imwrite("Difference.png", normMat);
	}
	else if (outChoice == 6) {
		cout << "Exiting..." << endl;
		exit(0);
	}
	else {
		cerr << "Incorrect input..exiting" << endl;
		exit(-1);
	}
}


/*********************************************************************************************
Description: Calculate the coherency matric from the feature vector
Author : Anupama Rajkumar
Input 1   : Feature vector
Input 2   : Feature vector number
Output 1  : Coherency Matrix from feature vector
**********************************************************************************************/

void Autoencoder::CalculateCoherencyMatrix(Mat& d, int row, Mat& coherencyMat) {

	coherencyMat.at<float>(0, 0) = exp(d.at<float>(row, 0));
	coherencyMat.at<float>(0, 1) = exp(d.at<float>(row, 3));
	coherencyMat.at<float>(0, 2) = exp(d.at<float>(row, 5));
	coherencyMat.at<float>(1, 0) = exp(d.at<float>(row, 3));
	coherencyMat.at<float>(1, 1) = exp(d.at<float>(row, 1));
	coherencyMat.at<float>(1, 2) = exp(d.at<float>(row, 7));
	coherencyMat.at<float>(2, 0) = exp(d.at<float>(row, 5));
	coherencyMat.at<float>(2, 1) = exp(d.at<float>(row, 7));
	coherencyMat.at<float>(2, 2) = exp(d.at<float>(row, 2));

}

/*********************************************************************************************
Description: Calculate the log of the coherency matrix in order to calculate the difference 
between input and reconstructed values
Author : Anupama Rajkumar
Input 1   : Coherency matrix
Output 1  : Log of coherency matrix
**********************************************************************************************/
void Autoencoder::CalculateLogMatrix(Mat& coherencyMat, Mat& aDash) {
	Mat eigenVal, eigenVec, eigenVecInv, logMatrix;
	eigenVal = Mat::zeros(3, 3, CV_32FC1);
	eigenVec = Mat::zeros(3, 3, CV_32FC1);
	eigenVecInv = Mat::zeros(3, 3, CV_32FC1);
	logMatrix = Mat::zeros(3, 3, CV_32FC1);
	/*step 1 : Calcuate eigenvalues of the matrix*/
	eigen(coherencyMat, eigenVal, eigenVec);

	invert(eigenVec, eigenVecInv);

	/*step 2: Logarithm of eigenvalues*/
	for (int r = 0; r < eigenVal.rows; r++) {
		for (int c = 0; c < eigenVal.cols; c++) {
			logMatrix.at<float>(r, c) = log(eigenVal.at<float>(r, c));
		}
	}

	/*multiply the result back*/
	mulSpectrums(eigenVec, logMatrix, aDash, 0, false);
	mulSpectrums(aDash, eigenVecInv, aDash, 0, false);

}

/*********************************************************************************************
Description: Calculate the difference between input and reconstructed values
Author : Anupama Rajkumar
Input 1   : Log of coherency matrix of input
Input 2   : Log of coherency matrix of reconstructed data
Output 1  : Difference 
**********************************************************************************************/
float Autoencoder::CalculateDifferenceMatrix(Mat& logInMatrix, Mat& logReconsMatrix) {
	float frobNorm = 0.;
	float sum = 0.;
	Mat diffOut = Mat::zeros(logInMatrix.size(), logInMatrix.type());
	/*calculate difference*/
	diffOut = logInMatrix - logReconsMatrix;
	/*calculate Frobenius norm*/
	for (int r = 0; r < diffOut.rows; r++) {
		for (int c = 0; c < diffOut.cols; c++) {
			sum += pow(diffOut.at<float>(r, c), 2);
		}
	}
	frobNorm = sqrt(sum);
	return frobNorm;
}


/*********************************************************************************************
Description: Function to prepare the calulated coherence matrix of entire image for a test/train 
data split that can be used for classification and calculation of accuracy
Author : Anupama Rajkumar
Input 1   : is the autoencoder vanilla or multilayer?
Input 2   : Input data
Input 3   : Hyperparameter k
Output 1  : Classification result in the form of a csv and image, performance metrics
**********************************************************************************************/
void Autoencoder::CalculateClassification(bool isAE, Data& data, Utils& utils, int k,
	Autoencoder& aeEncoder, KNN& knn, Performance& perform) {
	string perfFile;
	if (!isAE) {
		perfFile = "Performance_Metrics_SAE.csv";			//if stacked Autoencoder is used - future extension
	}
	else {
		perfFile = "Performance_Metrics_AE.csv";			//if vanilla autoencoder is used
	}

	Mat img;
	Mat classifiedImage;
	bool knnChoice = true;

	int rSize = 6640;
	int colSize = 1390;
	img = Mat::zeros(rSize, colSize, CV_32FC1);
	classifiedImage = Mat::zeros(rSize, colSize, CV_32FC1);

	ofstream perfPtr;
	perfPtr.open(perfFile, ofstream::out | ofstream::app);
	perfPtr << "PatchIdx" << "," << "OA" << endl;
	perfPtr.close();

	cout << "Enter knn choice..." << endl;
	cout << "Do you want OpenCV KNN ? Enter true/false" << endl;
	cout << "If true is chosen then opencv KNN will be used else custom KNN will be used" << endl;
	cin >> knnChoice;

	for (int patchIdx = 0; patchIdx < MAXNUMOFPATCHES; patchIdx++) {

		vector<pair<vector<Point2i>, uint>> patchPoint = utils.GetPatchPoints(patchIdx, data);
		vector<float> accuracy;
		string fileName, csvFile, classify;
		/*used for debugging -- comment out if not necessary*/
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
			/*start classification*/
			cout << " starting classification.." << endl;
			if (knnChoice) {
				cout << "Opencv KNN" << endl;
				knn.OpenCVKNNTest(trainVal, trainLabel, testVal, k, classResult);
			}
			else {
				cout << "Custom KNN" << endl;
				knn.KNNTest(trainVal, trainLabel, testVal, testLabel, k, classResult);
			}
			
		
			/*calculate prediction accuracy - performance metrics*/
			float intAccuracy = perform.calculatePredictionAccuracy(classResult, testLabel);
			accuracy.push_back(intAccuracy);
			cout << "Overall Accuracy for fold " << fold + 1 << " : " << intAccuracy << endl;

			/*store classification results*/
			for (int cnt = 0; cnt < data.testSamples.Samples.size(); cnt++) {
				Point2i newPoint(data.testSamples.Samples[cnt]);
				classifiedImage.at<int>(newPoint.x, newPoint.y) = (int)classResult[cnt];
			}

			/*releasing the memory for all the vectors*/
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
		/*calculate cross validation accuracy*/
		float avgAccuracy = 0.;
		float sumOfElements = 0.;
		for (auto& a : accuracy) {
			sumOfElements += a;
		}
		avgAccuracy = sumOfElements / 5;
		cout << "Sum :" << sumOfElements << endl;
		cout << "Average accuracy of patch " << patchIdx + 1 << ":" << avgAccuracy << endl;

		/*append performance data in a csv*/
		perfPtr.open(perfFile, ofstream::out | ofstream::app);
		perfPtr << patchIdx + 1 << "," << avgAccuracy << endl;
		perfPtr.close();

		/*store the classification result in a csv and generate corresponding image*/
		ofstream fileptr;
		fileptr.open(csvFile, ofstream::out);
		for (int row = 0; row < classifiedImage.rows; row++) {
			for (int col = 0; col < classifiedImage.cols; col++) {
				fileptr << classifiedImage.at<int>(row, col) << ",";
			}
			fileptr << endl;
		}
		utils.visualiseLabels(classifiedImage, fileName);
	}
}


