#include<iostream>
#include <opencv2/opencv.hpp>

#include "Autoencoder.h"


using namespace std;
using namespace cv;

/*
Written by : Anupama Rajkumar
Date : 25.06.2020
*/
Autoencoder::Autoencoder(int inputDim, int hiddenDim, double learningRate, double momentum) {
	m_dataDimension = inputDim;
	m_hiddenDimension = hiddenDim;
	m_learningRate = learningRate;
	m_momentum = momentum;

	m_inputValues.reserve(m_dataDimension);
	m_hiddenBias.reserve(m_hiddenDimension);
	m_hiddenBiasInit.reserve(m_hiddenDimension);
	m_hiddenBiasChanges.reserve(m_dataDimension);
	m_outputBias.reserve(m_dataDimension);
	m_outputBiasInit.reserve(m_dataDimension);
	m_outputBiasChanges.reserve(m_hiddenDimension * m_dataDimension);
	m_deltas.reserve(m_dataDimension);
	
	m_encoderWt.reserve(m_dataDimension * m_hiddenDimension);
	m_encoderWtInit.reserve(m_dataDimension * m_hiddenDimension);
	m_decoderWt.reserve(m_dataDimension * m_hiddenDimension);
	m_decoderWtInit.reserve(m_dataDimension * m_hiddenDimension);
	m_decoderWtChanges.reserve(m_dataDimension * m_hiddenDimension);
	m_updatedWt.reserve(m_dataDimension * m_hiddenDimension);
	m_encoderWtChanges.reserve(m_dataDimension * m_hiddenDimension);


	for (int cnt = 0; cnt < m_hiddenDimension; cnt++) {
		vector<float> val = this->random(m_dataDimension);
		m_encoderWtInit.push_back(val);
	}

	for (int cnt = 0; cnt < m_dataDimension; cnt++) {
		vector<float> val = this->random(m_hiddenDimension);
		m_decoderWtInit.push_back(val);
	}

	m_hiddenBiasInit = this->random(m_hiddenDimension);
	m_outputBiasInit = this->random(m_dataDimension);
}

void Autoencoder::InitializeWts() {
	
	m_encoderWt = m_encoderWtInit;
	m_decoderWt = m_decoderWtInit;

}

void Autoencoder::InitializeBias() {

	m_hiddenBias = m_hiddenBiasInit;
	m_outputBias = m_outputBiasInit;
}


vector<float> Autoencoder::random(size_t elementSize) {
	vector<float> result;
	for (size_t i = 0; i < elementSize; i++) {
		float val;
		val = ((float)rand() / (RAND_MAX));
		result.push_back(val);
	}
	return result;
}

Autoencoder::~Autoencoder() {

	m_encoderWt.clear();
	m_decoderWt.clear();
	m_hiddenBias.clear();
	m_hiddenBiasChanges.clear();
	m_outputBias.clear();
	m_outputBiasChanges.clear();
	m_outputBiasChanges.clear();
	m_encoderWtChanges.clear();
	m_decoderWtChanges.clear();
}

/*
Written by : Anupama Rajkumar
Date : 25.06.2020
*/
void Autoencoder::feedforward(vector<float>& m_hiddenValues, vector<float>& m_outputValues) {

	
	m_hiddenValues.reserve(m_hiddenDimension);
	m_outputValues.reserve(m_dataDimension);

	/*encoder - input->hidden layer*/
	for (auto i = 0; i < m_hiddenDimension; i++) {
		float total = 0.0;
		for (auto j = 0; j < m_dataDimension; j++) {
			total += m_encoderWt[i][j] * m_inputValues[j];	
		}
		/*assign this value to hiddenvalue after passing through activation	
		activation function used: sigmoid*/		
		total += m_hiddenBias[i];						//(W*x + b)
		m_hiddenValues.push_back(this->reLU(total));
	}

	/*decoder - hidden layer -> output*/
	for (auto i = 0; i < m_dataDimension; i++) {
		float total = 0.0;
		for (auto j = 0; j < m_hiddenDimension; j++) {
				total += m_decoderWt[i][j] * m_hiddenValues[j];		
		}
		/*assign this value to output after passing through activation
		activation function used: sigmoid*/
		total += m_outputBias[i];						//(W*x + b)
		m_outputValues.push_back(this->sigmoid(total));
	}

	/*printing output wt vector*/
	//cout << "output values are:" << endl;
	//this->PrintVector(m_outputValues);
}

/*
Modified by : Anupama Rajkumar
Date : 25.06.2020
*/
void Autoencoder::backpropagate(vector<float>& m_hiddenValues, vector<float>& m_outputValues) {

	/*for each output value - from outputlayer to hiddenlayer*/
	for (auto i = 0; i < m_dataDimension; i++) {
		vector<float> wtChanges;
		float delta = (m_outputValues[i] - m_inputValues[i])*this->sigmoidDerivation(m_outputValues[i]);
		m_deltas.push_back(delta);
		for (auto j = 0; j < m_hiddenDimension; j++) {
			/*adjusting weights vector from the hidden layer to output*/
			float changes = m_deltas[i] * m_hiddenValues[j];
			wtChanges.push_back(changes);
		}
		m_decoderWtChanges.push_back(wtChanges);
	}

	m_outputBiasChanges = m_deltas;
	
	/*from hidden layer to inputlayer*/
	for (auto i = 0; i < m_dataDimension; i++) {
		vector<float> wtUpdate;
		for (auto j = 0; j < m_hiddenDimension; j++) {
			 float changes = m_decoderWt[i][j] * m_deltas[i];
			 wtUpdate.push_back(changes);
		}
		m_updatedWt.push_back(wtUpdate);
	}

	for (auto i = 0; i < m_hiddenDimension; i++) {
		vector<float> wtChanges;
		for (auto j = 0; j < m_dataDimension; j++) {
			float dActivation = this->reLUDerivation(m_hiddenValues[i]);
			float changes = m_updatedWt[j][i] * dActivation * m_inputValues[j];		
			wtChanges.push_back(changes);
		}
		m_encoderWtChanges.push_back(wtChanges);
	}

	for (auto i = 0; i < m_hiddenDimension; i++) {
		float hBiaserr = 0.;
		for (auto j = 0; j < m_dataDimension; j++) {
			hBiaserr = m_encoderWt[i][j] * m_deltas[j];
		}
		hBiaserr = hBiaserr * this->sigmoidDerivation(m_hiddenValues[i]);
		m_hiddenBiasChanges.push_back(hBiaserr);
	}

	/*Adjusting the weights by SGD - encoder*/
	for (auto i = 0; i < m_hiddenDimension; i++) {
		vector<float> wtChange;
		for (auto j = 0; j < m_dataDimension; j++) {
			float weightChange = -(m_learningRate * m_momentum * m_encoderWtChanges[i][j]);
			float changes = m_encoderWt[i][j] + weightChange;
			wtChange.push_back(changes);
		}
		m_encoderWt.at(i) = wtChange;
	}

	/*Adjusting the weights by SGD - decoder*/
	for (auto i = 0; i < m_dataDimension; i++) {
		vector<float> wtChange;
		for (auto j = 0; j < m_hiddenDimension; j++) {
			float weightChange = -(m_learningRate * m_momentum * m_decoderWtChanges[i][j]);
			float changes = m_decoderWt[i][j] + weightChange;
			wtChange.push_back(changes);
		}
		m_decoderWt.at(i) = wtChange;
	}

	/*adjusting bias - outputBias*/
	for (auto i = 0; i < m_dataDimension; i++) {
		float biasChange = -(m_learningRate * m_momentum * m_outputBiasChanges[i]);
		float changes = m_outputBias[i] + biasChange;
		m_outputBias.at(i) = changes;
	}

	/*adjusting bias - hiddenBias*/
	for (auto i = 0; i < m_hiddenDimension; i++) {
		float biasChange = -(m_learningRate * m_momentum * m_hiddenBiasChanges[i]);
		float changes = m_hiddenBias[i] + biasChange;
		m_hiddenBias.at(i) = changes;
	}

	/*printing encoder wt vector*/
	/*cout << "encoder weights" << endl;
	for (int cnt = 0; cnt < m_encoderWt.size(); cnt++) {
		this->PrintVector(m_encoderWt[cnt]);
		cout << endl;
	}*/

	/*printing decoder wt vector*/
	/*cout << "decoder weights" << endl;
	for (int cnt = 0; cnt < m_decoderWt.size(); cnt++) {
		this->PrintVector(m_decoderWt[cnt]);
		cout << endl;
	}*/

	/*printing hidden bias*/
	//cout << "hidden bias values" << endl;
	//this->PrintVector(m_hiddenBias);

	//cout << "output bias values" << endl;
	//this->PrintVector(m_outputBias);
}


float Autoencoder::sigmoid(float d) {
	float sigmoidVal;
	float den = (1.0 + exp(-d));
	sigmoidVal = 1.0 / den;
	//cout << "den: " << den << " SigmoidVal: " << sigmoidVal << endl;
	return sigmoidVal;
}

float Autoencoder::sigmoidDerivation(float d) {
	return d * (1.0 - d);
}

void Autoencoder::train(vector<float>& data, vector<vector<float>>& m_OutputValuesF, int& epoch, int& cnt) {
	m_inputValues = data;
	vector<float> m_hiddenValues;
	vector<float> m_outputValues;
	this->feedforward(m_hiddenValues, m_outputValues);
	this->backpropagate(m_hiddenValues, m_outputValues);

	/*writing the output value to a map*/
	//m_OutputValuesF.at(cnt) = m_outputValues;
	if (epoch > 0) {
		m_OutputValuesF.at(cnt) = m_outputValues;
	}
	else {
		m_OutputValuesF.push_back(m_outputValues);
	}
	
}

void Autoencoder::test(vector<float>& data) {
	m_inputValues = data;
	vector<float> m_hiddenValues;
	vector<float> m_outputValues;
	this->feedforward(m_hiddenValues, m_outputValues);

	/*writing the output value to a map*/
	m_OutputValuesF.push_back(m_outputValues);
}

void Autoencoder::PrintVector(vector<float>& data) {
	for (int cnt = 0; cnt < data.size(); cnt++) {
		cout << data[cnt] << " ";
		cout << endl;
	}
	cout << endl;
}

float Autoencoder::reLU(float d) {
	if (d < 0) {
		return 0;
	}
	else {
		return d;
	}
}

float Autoencoder::reLUDerivation(float d) {
	if (d < 0) {
		return 0;
	}
	else {
		return 1;
	}
}

