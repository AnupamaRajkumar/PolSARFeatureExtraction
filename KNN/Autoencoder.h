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

	//void train(vector<double>& data);
	//void test(vector<double>& data);

	double *random(size_t elementSize);
	double sigmoid(double value);
	double sigmoidDerivation(double value);

private:
	int m_dataDimension;				// #of output neurons = #of input neurons
	int m_hiddenDimension;
	double m_learningRate;
	double m_momentum;

	vector<double> m_inputValues;
	vector<double> m_hiddenValues;
	vector<double> m_outputValues;

	double **m_encoderWt;
	double **m_decoderWt;
	double **m_updatedWt;
	double **m_encoderWtChanges;
	double **m_decoderWtChanges;

	double *m_inputBias;
	double *m_deltas;

	void feedforward();
	void backpropagate();

};


#endif
