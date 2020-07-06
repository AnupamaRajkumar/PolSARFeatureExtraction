#pragma once
#ifndef _AUTOENCODER_
#define _AUTOENCODER_

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

//referred from : https://github.com/turkdogan/autoencoder

//cuda example in : https://github.com/lostleaf/cuda-autoencoder

class Autoencoder {

public:
	Autoencoder(int inputDim, int hiddenDim, double learningRate, double momentum);
	~Autoencoder();

	void train(vector<float>& data, vector<vector<float>>& m_OutputValuesF, int& epoch, int& cnt);
	void test(vector<float>& data);

	void PrintVector(vector<float>& data);
	void InitializeWts();
	void InitializeBias();

	vector<float> random(size_t elementSize);
	float sigmoid(float value);
	float sigmoidDerivation(float value);
	float reLU(float value);
	float reLUDerivation(float value);

private:
	int m_dataDimension;				// #of output neurons = #of input neurons
	int m_hiddenDimension;
	double m_learningRate;
	double m_momentum;

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
	vector<vector<float>> m_decoderWt;
	vector<vector<float>> m_decoderWtInit;
	vector<vector<float>> m_updatedWt;
	vector<vector<float>> m_encoderWtChanges;
	vector<vector<float>> m_decoderWtChanges;
	vector<vector<float>> m_OutputValuesF;


	void feedforward(vector<float>& m_hiddenValues, vector<float>& m_outputValues);
	void backpropagate(vector<float>& m_hiddenValues, vector<float>& m_outputValues);

};


#endif
