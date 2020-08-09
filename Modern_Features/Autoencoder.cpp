#include<iostream>
#include <opencv2/opencv.hpp>

#include "Autoencoder.h"


using namespace std;
using namespace cv;


Autoencoder::Autoencoder() {
	/*do nothing*/
}

/******************************************************************************
Paramterised constructor of Autoencoder (AE)
Author : Anupama Rajkumar
Description: This function initialises the values of various variables used
Input 1 : Input dimension of the AE
Input 2 : Hidden/encoder dimension of AE
Input 3	: Learning Rate 
Input 4	: Momentum 
*******************************************************************************/
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
	m_updatedWt.reserve(m_dataDimension * m_hiddenDimension);
	m_encoderWtChanges.reserve(m_dataDimension * m_hiddenDimension);

	/*initialise the encoder weights to random values*/
	for (int cnt = 0; cnt < m_hiddenDimension; cnt++) {
		vector<float> val = this->random(m_dataDimension);
		m_encoderWtInit.push_back(val);
	}

	/*initialise the biases to random initial values*/
	m_hiddenBiasInit = this->random(m_hiddenDimension);
	m_outputBiasInit = this->random(m_dataDimension);


}

/******************************************************************************
Initilising the intermediate variables of AE used for weight optimisation
Author : Anupama Rajkumar
Description: This function ensures that weight parameters are initialised to same
random value for every pixel and intermediate weight variables are reset
*******************************************************************************/

void Autoencoder::InitializeWts() {	
	m_encoderWt = m_encoderWtInit;
	vector<float> val;
	vector < vector<float> > dD, hDim;
	for (int cnt = 0; cnt < m_dataDimension; cnt++) {
		val.push_back(0);
	}
	m_deltas = val;
	for (int i = 0; i < m_dataDimension; i++) {
		vector<float> hD;
		for (int j = 0; j < m_hiddenDimension; j++) {
			hD.push_back(0);
		}
		dD.push_back(hD);
	}
	m_updatedWt = dD;

	for (int i = 0; i < m_hiddenDimension; i++) {
		vector<float> dDim;
		for (int j = 0; j < m_dataDimension; j++) {
			dDim.push_back(0);
		}
		hDim.push_back(dDim);
	}
	m_encoderWtChanges = hDim;
	v = 0.;
}

/******************************************************************************
Initilising the intermediate variables of AE used for bias optimisation
Author : Anupama Rajkumar
Description: This function ensures that bias parameters are initialised to same
random value for every pixel and intermediate bias variables are reset
*******************************************************************************/
void Autoencoder::InitializeBias() {
	m_hiddenBias = m_hiddenBiasInit;
	m_outputBias = m_outputBiasInit;
	vector<float> val;
	for (int cnt = 0; cnt < m_hiddenDimension; cnt++) {
		val.push_back(0);
	}
	m_hiddenBiasChanges = val;
}


/******************************************************************************
Random weight generator
Author : Anupama Rajkumar
Description: This function generates random values to initialise weight and bias
Input 1 : The dimension of the vector to be initialised randomly
Output	: Vector initialised with random values
*******************************************************************************/

vector<float> Autoencoder::random(size_t elementSize) {
	vector<float> result;
	for (size_t i = 0; i < elementSize; i++) {
		float val;
		val = ((float)rand() / (RAND_MAX));
		result.push_back(val);
	}
	return result;
}

/******************************************************************************
Destructor of Autoencoder (AE)
Author : Anupama Rajkumar
Description: This function clears the memory held by various variables
*******************************************************************************/
Autoencoder::~Autoencoder() {

	m_encoderWt.clear();
	m_hiddenBias.clear();
	m_hiddenBiasChanges.clear();
	m_outputBias.clear();
	m_outputBiasChanges.clear();
	m_encoderWtChanges.clear();
	m_updatedWt.clear();
	m_deltas.clear();
}

/******************************************************************************
Feedforward network of Autoencoder (AE)
Author : Anupama Rajkumar
Description: Feedforward implementation of AE as per the formula (W*x + b)
Output 1 : The hidden/encoder layer values
Output 2 : The output layer value calculated from weight and bias parameters
*******************************************************************************/
void Autoencoder::feedforward(vector<float>& m_hiddenValues, vector<float>& m_outputValues) {

	
	m_hiddenValues.reserve(m_hiddenDimension);
	m_outputValues.reserve(m_dataDimension);

	/*encoder - input -> hidden layer*/
	for (auto i = 0; i < m_hiddenDimension; i++) {
		float total = 0.0;
		for (auto j = 0; j < m_dataDimension; j++) {
			total += m_encoderWt[i][j] * m_inputValues[j];	
		}
		/*assign this value to hiddenvalue after passing through activation	
		activation function used: linear*/		
		total += m_hiddenBias[i];						//(W*x + b)
		m_hiddenValues.push_back(this->sigmoid(total));
	}

	/*decoder - hidden layer -> output*/
	for (auto i = 0; i < m_dataDimension; i++) {
		float total = 0.0;
		for (auto j = 0; j < m_hiddenDimension; j++) {
				total += m_encoderWt[j][i] * m_hiddenValues[j];			//decoder wt is transpose of encoder wt
		}
		/*assign this value to output after passing through activation
		activation function used: sigmoid*/
		total += m_outputBias[i];						//(W'*x + b)
		m_outputValues.push_back(total);
	}
}

/******************************************************************************
Backpropagatation algorithm to adjust weight and bias parameters of AE
Author : Anupama Rajkumar
Description: Feedforward implementation of AE as per the formula (W*x + b)
Input 1		: Hidden value calculated in feedforward step
Input 2		: Output value calculated in feedforward step
Output 1	: Optimised weight parameter
Output 2	: Optimised bias parameter
*******************************************************************************/
void Autoencoder::backpropagate(vector<float>& m_hiddenValues, vector<float>& m_outputValues) {

	/*for each output value - from outputlayer to hiddenlayer*/
	for (auto i = 0; i < m_dataDimension; i++) {
		vector<float> wtChanges;
		float delta = (m_outputValues[i] - m_inputValues[i])*m_outputValues[i];		
		m_deltas.at(i) = delta;
	}

	m_outputBiasChanges = m_deltas;
	
	/*from hidden layer to inputlayer*/
	for (auto i = 0; i < m_dataDimension; i++) {
		vector<float> wtUpdate;
		for (auto j = 0; j < m_hiddenDimension; j++) {
			 float changes = m_encoderWt[j][i] * m_deltas[i];
			 wtUpdate.push_back(changes);
		}
		m_updatedWt.at(i) = wtUpdate;
	}
	/*calculate weight changes*/
	for (auto i = 0; i < m_hiddenDimension; i++) {
		vector<float> wtChanges;
		for (auto j = 0; j < m_dataDimension; j++) {
			float dActivation = this->sigmoidDerivation(m_hiddenValues[i]);
			float changes = m_updatedWt[j][i] * dActivation * m_inputValues[j];		
			wtChanges.push_back(changes);
		}
		m_encoderWtChanges.at(i) = wtChanges;
	}
	/*calculate bias changes*/
	for (auto i = 0; i < m_hiddenDimension; i++) {
		float hBiaserr = 0.;
		for (auto j = 0; j < m_dataDimension; j++) {
			hBiaserr = m_encoderWt[i][j] * m_deltas[j];
		}
		hBiaserr = hBiaserr * this->sigmoidDerivation(m_hiddenValues[i]);
		m_hiddenBiasChanges.at(i) = hBiaserr;
	}

	/*Adjusting the weights by SGD - encoder*/
	for (auto i = 0; i < m_hiddenDimension; i++) {
		vector<float> wtChange;
		for (auto j = 0; j < m_dataDimension; j++) {
			v = m_momentum * v + (1 - m_momentum) * m_encoderWtChanges[i][j];
			float weightChange = -(m_learningRate * v);
			float changes = m_encoderWt[i][j] + weightChange;
			wtChange.push_back(changes);
		}
		m_encoderWt.at(i) = wtChange;
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

}

/******************************************************************************
Sigmoid activation function
Author : Anupama Rajkumar
Description: Implementation of activation function - Sigmoid
Input 1		: Input data
Output 1	: Sigmoid value of input value
*******************************************************************************/
float Autoencoder::sigmoid(float d) {
	float sigmoidVal;
	float den = (1.0 + exp(-d));
	sigmoidVal = 1.0 / den;
	//cout << "den: " << den << " SigmoidVal: " << sigmoidVal << endl;
	return sigmoidVal;
}

/******************************************************************************
Derivative of sigmoid
Author : Anupama Rajkumar
Description: Sigmoid derivative used in backpropagation
Input 1		: Input data
Output 1	: Sigmoid derivative of input value
*******************************************************************************/
float Autoencoder::sigmoidDerivation(float d) {
	return d * (1.0 - d);
}

/******************************************************************************
reLU activation function
Author : Anupama Rajkumar
Description: Implementation of activation function - reLU
Input 1		: Input data
Output 1	: reLU value of input value
*******************************************************************************/

float Autoencoder::reLU(float d) {
	if (d < 0) {
		return 0;
	}
	else {
		return d;
	}
}


/******************************************************************************
Derivative of reLU
Author : Anupama Rajkumar
Description: reLU derivative used in backpropagation
Input 1		: Input data
Output 1	: reLU derivative of input value
*******************************************************************************/
float Autoencoder::reLUDerivation(float d) {
	if (d < 0) {
		return 0;
	}
	else {
		return 1;
	}
}


/******************************************************************************
Training the input feature vector to extract features
Author : Anupama Rajkumar
Description: This function is used to extract feature from the input feature vector
from each pixel in the image
Input 1		: Input feature vector
Input 2		: The current epoch count
Input 3		: Total number of epochs
Output 1	: Learned feature vector
Output 2	: The output vector
*******************************************************************************/
void Autoencoder::train(vector<float>& data, int& cnt, int& epoch) {
	m_inputValues = data;
	vector<float> m_hiddenValues;
	vector<float> m_outputValues;
	this->feedforward(m_hiddenValues, m_outputValues);
	this->backpropagate(m_hiddenValues, m_outputValues);
	/*writing the output value to a map*/
	if (epoch > 0) {
		m_featureVector.at(cnt) = m_hiddenValues;
		m_outputVector.at(cnt) = m_outputValues;
	}
	else {
		m_featureVector.push_back(m_hiddenValues);
		m_outputVector.push_back(m_outputValues);
	}

	float loss = 0.;
#if 0
	cout << "Loss values" << endl;
	for (int cnt = 0; cnt < m_dataDimension; cnt++) {
		loss += 0.5*pow((m_inputValues[cnt] - m_outputValues[cnt]),2);
	}
	cout << loss << endl;
#endif
}


/******************************************************************************
Helper function to print a vector
Author : Anupama Rajkumar
Description: Print a vector
Input 1		: Input vector
*******************************************************************************/
void Autoencoder::PrintVector(vector<float>& data) {
	for (int cnt = 0; cnt < data.size(); cnt++) {
		cout << data[cnt] << " ";
		cout << endl;
	}
	cout << endl;
}

/******************************************************************************
Helper function to save a encoder and bias parameters
Author : Anupama Rajkumar
Description: Save optimised encoder and bias parameters. Can be used for stacked
autoencoders in order to reconstrct final output values
Output 1	: optimised weight vector
Output 2	: optimised bias vector
*******************************************************************************/
void Autoencoder::SaveParameters(vector<vector<float>>& encoderWt, vector<float>& outputBias)
{
	encoderWt = m_encoderWt;
	outputBias = m_outputBias;
}

/******************************************************************************
Reconstructing the output values for stacked AE
Author : Anupama Rajkumar
Description: This function uses the optimised weight and bias parameters from 
previous layers in a multilayer or stacked AE to calculate the output value
Input 1	 : optimised weight vector
Input 2	 : optimised bias vector
Input 3  : Input vector
Output 1 : Reconstructed output
*******************************************************************************/
void Autoencoder::ReconstructOutput(vector<vector<float>>& encoderWt, vector<float>& outputBias, 
									vector<float>& input, vector<float>& output){
	output.clear();
	for (auto i = 0; i < m_dataDimension; i++) {
		float total = 0.0;
		for (auto j = 0; j < m_hiddenDimension; j++) {
			total += encoderWt[j][i] * input[j];			//decoder wt is transpose of encoder wt
		}
		/*assign this value to output after passing through activation
		activation function used: sigmoid*/
		total += outputBias[i];						//(W'*x + b)
		output.push_back(total);
	}
}

