#include<iostream>
#include <opencv2/opencv.hpp>

#include "Autoencoder.h"


using namespace std;
using namespace cv;

/*
Modified by : Anupama Rajkumar
Date : 25.06.2020
*/
Autoencoder::Autoencoder(int inputDim, int hiddenDim, double learningRate, double momentum) {
	m_dataDimension = inputDim;
	m_hiddenDimension = hiddenDim;
	m_learningRate = learningRate;
	m_momentum = momentum;

	m_inputValues = vector<double>();
	m_hiddenValues = vector<double>();
	m_outputValues = vector<double>();

	m_encoderWt = new double*[m_hiddenDimension];
	for (int cnt = 0; cnt < m_hiddenDimension; cnt++) {
		m_encoderWt[cnt] = this->random(m_dataDimension);
	}

	m_decoderWt = new double*[m_dataDimension];	
	for (int cnt = 0; cnt < m_dataDimension; cnt++) {
		m_decoderWt[cnt] = this->random(m_hiddenDimension);
	}

	m_decoderWtChanges = new double*[m_hiddenDimension];
	for (int cnt = 0; cnt < m_dataDimension; cnt++) {
		m_decoderWtChanges[cnt] = new double[m_dataDimension]();
	}
	m_updatedWt = new double*[m_hiddenDimension];
	for (int cnt = 0; cnt < m_dataDimension; cnt++) {
		m_updatedWt[cnt] = new double[m_dataDimension]();
	}

	m_encoderWtChanges = new double*[m_dataDimension];
	for (int cnt = 0; cnt < m_hiddenDimension; cnt++) {
		m_encoderWtChanges[cnt] = new double[m_hiddenDimension]();
	}


	m_inputBias = new double[m_hiddenDimension];			/*think of the dimension of the bias*/
	m_deltas = new double[m_dataDimension]();
}

double* Autoencoder::random(size_t elementSize) {
	double *result = new double[elementSize];
	for (size_t i = 0; i < elementSize; i++) {
		result[i] = ((double)rand() / (RAND_MAX));
	}
	return result;
}

Autoencoder::~Autoencoder() {

	for (auto i = 0; i < m_hiddenDimension; i++)
	{
		delete[] m_encoderWt[i];
		delete[] m_encoderWtChanges[i];
	}

	for (auto i = 0; i < m_dataDimension; i++)
	{
		delete[] m_decoderWt[i];
		delete[] m_decoderWtChanges[i];
	}
	delete[] m_deltas;
}

/*
Modified by : Anupama Rajkumar
Date : 25.06.2020
*/
void Autoencoder::feedforward() {

	/*encoder - input->hidden layer*/
	for (auto i = 0; i < m_hiddenDimension; i++) {
		double total = 0.0;
		for (auto j = 0; j < m_dataDimension; j++) {
			total += m_encoderWt[i][j] * m_inputValues[j];
		}
		/*assign this value to hiddenvalue after passing through activation
		activation function used: sigmoid*/
		/*todo - add bias*/
		m_hiddenValues[i] = this->sigmoid(total);
	}

	/*decoder - hidden layer -> output*/
	for (auto i = 0; i < m_dataDimension; i++) {
		double total = 0.0;
		for (auto j = 0; j < m_hiddenDimension; j++) {
			total += m_decoderWt[i][j] * m_hiddenValues[j];
		}
		/*assign this value to output after passing through activation
		activation function used: sigmoid*/
		m_outputValues[i] = this->sigmoid(total);
	}
}

/*
Modified by : Anupama Rajkumar
Date : 25.06.2020
*/
void Autoencoder::backpropagate() {
	/*for each output value - from outputlayer to hiddenlayer*/
	for (auto i = 0; i < m_dataDimension; i++) {
		m_deltas[i] = (m_outputValues[i] - m_inputValues[i])*this->sigmoidDerivation(m_outputValues[i]);
		for (auto j = 0; j < m_hiddenDimension; j++) {
			/*adjusting weights vector from the hidden layer to output*/
			m_decoderWtChanges[i][j] = m_deltas[i]*m_hiddenValues[j] ;
		}
	}
	
	/*from hidden layer to inputlayer*/
	for (auto i = 0; i < m_dataDimension; i++) {
		for (auto j = 0; j < m_hiddenDimension; j++) {
			m_updatedWt[i][j] = m_decoderWt[i][j] * m_deltas[i];
		}
	}

	for (auto i = 0; i < m_hiddenDimension; i++) {
		for (auto j = 0; j < m_dataDimension; j++) {
			double dActivation = this->sigmoidDerivation(m_hiddenValues[i]);
			m_encoderWtChanges[i][j] = m_updatedWt[j][i] * dActivation * m_inputValues[j];
		}
	}

	/*Adjusting the weights - encoder*/
	for (auto i = 0; i < m_hiddenDimension; i++) {
		for (auto j = 0; j < m_dataDimension; j++) {
			double weightChange = -(m_learningRate * m_momentum * m_encoderWtChanges[i][j]);
			m_encoderWt[i][j] += weightChange;
		}
	}

	/*Adjusting the weights - decoder*/
	for (auto i = 0; i < m_dataDimension; i++) {
		for (auto j = 0; j < m_hiddenDimension; j++) {
			double weightChange = -(m_learningRate * m_momentum * m_decoderWtChanges[i][j]);
			m_decoderWt[i][j] += weightChange;
		}
	}
}





double Autoencoder::sigmoid(double d) {
	return 1.0 / (1.0 + exp(-d));
}

double Autoencoder::sigmoidDerivation(double d) {
	return d * (1.0 - d);
}