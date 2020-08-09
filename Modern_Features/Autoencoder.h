#pragma once
#ifndef _AUTOENCODER_
#define _AUTOENCODER_

#include <iostream>
#include <opencv2/opencv.hpp>


#include "Data.h"
#include "Utils.h"
#include "KNN.h"
#include "Performance.h"

using namespace std;
using namespace cv;


class Autoencoder {

public:
	Autoencoder();
	Autoencoder(int inputDim, int hiddenDim, double learningRate, double momentum);
	~Autoencoder();

	void train(vector<float>& data, int& cnt, int& epoch);
	void PrintVector(vector<float>& data);
	void InitializeWts();
	void InitializeBias();
	void SaveParameters(vector<vector<float>>& encoderWt, vector<float>& outputBias);
	void ReconstructOutput(vector<vector<float>>& encoderWt, vector<float>& outputBias, vector<float>& input,
						   vector<float>& output);

	vector<float> random(size_t elementSize);
	float sigmoid(float value);
	float sigmoidDerivation(float value);
	float reLU(float value);
	float reLUDerivation(float value);
	vector<vector<float>> m_featureVector;
	vector<vector<float>> m_outputVector;

	/*Autoencoder operation menu*/
	void AutoencoderUserMenu(vector<vector<float>>& coherenceVec, Data& data);
	void CalculateClassification(bool isAE, Data& data, Utils& utils, int k,
		Autoencoder& aeEncoder, KNN& knn, Performance& perform);
	float CalculateDifferenceMatrix(Mat& logInMatrix, Mat& logReconsMatrix);
	void CalculateCoherencyMatrix(Mat& d, int row, Mat& coherencyMat);
	void CalculateLogMatrix(Mat& coherencyMat, Mat& logMatrix);
private:
	int m_dataDimension;				// #of output neurons = #of input neurons
	int m_hiddenDimension;
	double m_learningRate;
	double m_momentum;
	double v;

	vector<float> m_inputValues;
	vector<float> m_hiddenBias;
	vector<float> m_hiddenBiasInit;
	vector<float> m_outputBias;	
	vector<float> m_outputBiasInit;
	vector<float> m_hiddenBiasChanges;
	vector<float> m_outputBiasChanges;
	vector<float> m_deltas;


	vector<vector<float>> m_encoderWt;
	vector<vector<float>> m_encoderWtInit;
	vector<vector<float>> m_updatedWt;
	vector<vector<float>> m_encoderWtChanges;


	void feedforward(vector<float>& m_hiddenValues, vector<float>& m_outputValues);
	void backpropagate(vector<float>& m_hiddenValues, vector<float>& m_outputValues);

};


#endif
