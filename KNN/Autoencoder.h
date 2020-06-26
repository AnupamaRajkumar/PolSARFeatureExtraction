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

	void train(vector<float>& data);
	void test(vector<float>& data);

	float *random(size_t elementSize);
	float sigmoid(float value);
	float sigmoidDerivation(float value);

private:
	int m_dataDimension;				// #of output neurons = #of input neurons
	int m_hiddenDimension;
	double m_learningRate;
	double m_momentum;

	vector<float> m_inputValues;
	vector<float> m_hiddenValues;
	vector<float> m_outputValues;

	float **m_encoderWt;
	float **m_decoderWt;
	float **m_updatedWt;
	float **m_encoderWtChanges;
	float **m_decoderWtChanges;

	float *m_inputBias;
	float *m_deltas;

	void feedforward();
	void backpropagate();

};


#endif
