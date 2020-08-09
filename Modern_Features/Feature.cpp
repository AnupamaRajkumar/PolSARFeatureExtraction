/**************************************************
Step 1: Form RGB Image (Eg use Pauli decomposition)
Step 2: Oversegmentation
Step 3: Extract feature vectors based on RGB Image 
Step 4: Generate a new probablistic metric by applying softmax
Step 5: Introduce this to KNN to improve accuracy of classification

****************************************************/

#include "Feature.h"
#include "Data.h"
#include "Utils.h"
#include "cvFeatures.hpp"

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>




using namespace std;
using namespace cv;


/***************************************************************************
Description: Calculate Lexi decomposition
Input 1 : hh matrix for all pixels in the entire image
Input 2 : vv matrix for all pixels in the entire image
Input 3 : hv matrix for all pixels in the entire image
Output  : Lexi decomposition vector for all pixels in the entire image
****************************************************************************/
void Feature::getLexiBasis(const Mat& hh, const Mat& vv, const Mat& hv, vector<Mat>& lexi) {
	lexi.push_back(hh);
	lexi.push_back(sqrt(2.0) * hv);
	lexi.push_back(vv);
}

/***************************************************************************
Description: Calculate Pauli decomposition
Author : Anupama Rajkumar
Input 1 : hh matrix for all pixels in the entire image
Input 2 : vv matrix for all pixels in the entire image
Input 3 : hv matrix for all pixels in the entire image
Output  : Pauli decomposition vector for all pixels in the entire image
****************************************************************************/
void Feature::getPauliBasis(const Mat& hh, const Mat& vv, const Mat& hv, vector<Mat>& pauli) {
	pauli.push_back((hh + vv) / sqrt(2.0));				//blue
	pauli.push_back((hh - vv) / sqrt(2.0));				//red
	pauli.push_back(hv * sqrt(2.0));					//green
}

/***************************************************************************
Description: Calculate amplitude of complex matrix 
Author : Jun Xiang
Input 1 : Input complex matrix
Output  : Amplitude of complex matrix calculated as sqrt(re*re + im*im)
****************************************************************************/
Mat Feature::getComplexAmpl(const Mat& in) {

	vector<Mat> channels;

	split(in, channels);
	pow(channels[0], 2, channels[0]);
	pow(channels[1], 2, channels[1]);
	Mat out = channels[0] + channels[1];
	pow(out, 0.5, out);

	return out;

}

/***************************************************************************
Description: Calculate log transform of input matrix
Author : Jun Xiang
Input 1 : Input matrix
Output  : log of the input matrix
****************************************************************************/
Mat Feature::logTransform(const Mat& in) {

	Mat out;
	if (in.channels() == 2) {
		out = getComplexAmpl(in);
	}
	else
		out = in;

	out = out + 1;
	log(out, out);

	return out;

}

/***************************************************************************
Description: Calculate log transform of input vector - overridden
Author : Anupama Rajkumar
Input 1 : Input vector
Output  : log of the input vector
****************************************************************************/
vector<float> Feature::logTransform(vector<float>& in) {

	vector<float> out;
	out = in;
	log(out, out);
	return out;
}

/***************************************************************************
Description: Calculate coherence matrix from pauli decomposition
Input 1 : Pauli decomposition vector
Input 2 : Blurring filter window size
Output  : Coherency Matrix
****************************************************************************/

void Feature::GetCoherencyMat(vector<Mat>& pauli, int winSize, vector<Mat>& coherencyMat) {
	this->vec2mat(pauli, winSize, coherencyMat);
}

/***************************************************************************
Description: Calculate coherence matrix from pauli decomposition
Modified by : Anupama Rajkumar
Input 1 : Pauli decomposition vector
Input 2 : Blurring filter window size
Output  : Coherency Matrix for entire image
****************************************************************************/
void Feature::vec2mat(const vector<Mat>& basis, int winSize, vector<Mat>& mat) {
	Mat m00, m01, m02, m11, m12, m22;

	mulSpectrums(basis.at(0), basis.at(0), m00, 0, true); //|k_0 | ^ 2
	mulSpectrums(basis.at(0), basis.at(1), m01, 0, true); //k_0*conj(k_1)
	mulSpectrums(basis.at(0), basis.at(2), m02, 0, true); //k_0*conj(k_2)
	mulSpectrums(basis.at(1), basis.at(1), m11, 0, true); //k_1|^2
	mulSpectrums(basis.at(1), basis.at(2), m12, 0, true); //k_1*conj(k_2)
	mulSpectrums(basis.at(2), basis.at(2), m22, 0, true); //|k_2|^2 

	/*blur out noise*/
	cv::blur(m00, m00, Size(winSize, winSize));
	cv::blur(m01, m01, Size(winSize, winSize));
	cv::blur(m02, m02, Size(winSize, winSize));
	cv::blur(m11, m11, Size(winSize, winSize));
	cv::blur(m12, m12, Size(winSize, winSize));
	cv::blur(m22, m22, Size(winSize, winSize));

	vector<Mat> m00Comp, m11Comp, m22Comp, m01Comp, m02Comp, m12Comp;
	split(m00, m00Comp);
	split(m11, m11Comp);
	split(m22, m22Comp);
	split(m01, m01Comp);
	split(m02, m02Comp);
	split(m12, m12Comp);

	mat.push_back(m00Comp[0]);			//diagonal element m00
	mat.push_back(m11Comp[0]);			//diagonal element m11
	mat.push_back(m22Comp[0]);			//diagonal element m22
	mat.push_back(abs(m01Comp[0]));		//real component m01
	mat.push_back(abs(m01Comp[1]));		//imag component m01
	mat.push_back(abs(m02Comp[0]));		//real component m02
	mat.push_back(abs(m02Comp[1]));		//imag component m02
	mat.push_back(abs(m12Comp[0]));		//real component m12
	mat.push_back(abs(m12Comp[1]));		//imag component m12

}


/**********************************************************************************************
Description: helper function to convert input vector of matrix to vector of vector
Modified by : Anupama Rajkumar
Input 1 : vector of input matrix
Output  : vector of vector of input matrix
**********************************************************************************************/
void ConvertToVector(vector<Mat>& coherencyMat, vector<vector<float>>& coherencyVec) {
	for (int cnt = 0; cnt < coherencyMat.size(); cnt++) {
		vector<float> vec;
		vec.assign(coherencyMat[cnt].begin <float>(), coherencyMat[cnt].end<float>());
		coherencyVec.push_back(vec);
	}
}


/*********************************************************************************************
Description: helper function to convert input matrix to output vector - overriden
Modified by : Anupama Rajkumar
Input 1 : Input Matrix
Output  : Output Vector
**********************************************************************************************/
void ConvertToVector(Mat& coherencyMat, vector<unsigned char>& coherencyVec) {
		vector<unsigned char> vec;
		vec.assign(coherencyMat.begin <unsigned char>(), coherencyMat.end<unsigned char>());
		coherencyVec = vec;
}


/*********************************************************************************************
Description: This function calculates coherency matrix for entire image
Modified by : Anupama Rajkumar
Input 1   : Input Data
Output 1  : Coherency vector for entire image
Output 2  : Corresponding label vector for entire image
**********************************************************************************************/
void Feature::GetCoherencyFeatures(Data data, vector<vector<float>>& resultVec, vector<unsigned char>& classValue) {
	Mat hh, vv, hv;
	Utils util;
	int winSize = 3;
	hh = data.data[0];
	vv = data.data[1];
	hv = data.data[2];

	vector<vector<double>> pauliVec;
	vector<Mat> pauli;
	/*calculate pauli decomposition*/
	this->getPauliBasis(hh, vv, hv, pauli);
	vector<Mat> coherencyMat;	
	vector<vector<float>> coherencyVec;
	Mat labelMap;
	labelMap = Mat::zeros(data.labelImages[0].size(), CV_8UC1);
	/*generate coherency matrix*/
	this->GetCoherencyMat(pauli, winSize, coherencyMat);

	ConvertToVector(coherencyMat, coherencyVec);	
	copy(coherencyVec.begin(), coherencyVec.end(), back_inserter(resultVec));
	/*generate label value for each pixel*/
	util.generateLabelMap(data.labelImages, labelMap);
	ConvertToVector(labelMap, classValue);
	/*take log of the coherence matrix values*/
	for (auto& e : resultVec) {
		vector<float> temp;
		temp = this->logTransform(e);

		e = temp;
	}
}





