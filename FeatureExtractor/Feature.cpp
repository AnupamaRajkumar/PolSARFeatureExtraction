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

//#if 0



using namespace std;
using namespace cv;


/************************************************
Pauli Decomposition
************************************************/
void Feature::lexi2pauli(vector<Mat>& lexi, vector<Mat>& pauli) {

	Mat k1, k2, k3;

	// k_1 = sqrt(1/2) * (s_hh + s_vv)
	k1 = sqrt(0.5)*(lexi.at(0) + lexi.at(1));  //same as alpha

	// k_2 = sqrt(1/2) * (s_hh - s_vv)
	k2 = sqrt(0.5)*(lexi.at(0) - lexi.at(1));	//same as beta

	// k_1 = sqrt(1/2) * 2 * s_hv
	if (lexi.size() == 3)
		k3 = 2 * sqrt(0.5)*lexi.at(2);			//same as gamma

	pauli.push_back(k1);
	pauli.push_back(k2);
	if (lexi.size() == 3)
		pauli.push_back(k3);
}

/************************************************
Lexi Decomposition
************************************************/
void Feature::getLexiBasis(const Mat& hh, const Mat& vv, const Mat& hv, vector<Mat>& lexi) {
	lexi.push_back(hh);
	lexi.push_back(sqrt(2.0) * hv);
	lexi.push_back(vv);
}

void Feature::getPauliBasis(const Mat& hh, const Mat& vv, const Mat& hv, vector<Mat>& pauli) {
	pauli.push_back((hh + vv) / sqrt(2.0));
	pauli.push_back((hh - vv) / sqrt(2.0));
	pauli.push_back(hv * sqrt(2.0));
}



/*Author : Jun Xiang*/
Mat Feature::getComplexAmpl(const Mat& in) {

	vector<Mat> channels;

	split(in, channels);
	pow(channels[0], 2, channels[0]);
	pow(channels[1], 2, channels[1]);
	Mat out = channels[0] + channels[1];
	pow(out, 0.5, out);

	return out;

}

/*Author : Jun Xiang*/
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

vector<float> Feature::logTransform(vector<float>& in) {

	vector<float> out;
	out = in;
	log(out, out);

	return out;

}


void Feature::GetCoherencyMat(vector<Mat>& pauli, vector<Mat>& coherencyMat, int winSize) {
	this->vec2mat(pauli, coherencyMat, winSize);
}


void Feature::vec2mat(const vector<Mat>& basis, vector<Mat>& mat, int winSize) {
	Mat m00, m01, m02, m11, m12, m22;

	mulSpectrums(basis.at(0), basis.at(0), m00, 0, true); //|k_0 | ^ 2
	mulSpectrums(basis.at(0), basis.at(1), m01, 0, true); //k_0*conj(k_1)
	mulSpectrums(basis.at(0), basis.at(2), m02, 0, true); //k_0*conj(k_2)
	mulSpectrums(basis.at(1), basis.at(1), m11, 0, true); //k_1|^2
	mulSpectrums(basis.at(1), basis.at(2), m12, 0, true); //k_1*conj(k_2)
	mulSpectrums(basis.at(2), basis.at(2), m22, 0, true); //|k_2|^2 

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

	mat.push_back(m00Comp[0]);		//diagonal element m00
	mat.push_back(m11Comp[0]);		//diagonal element m11
	mat.push_back(m22Comp[0]);		//diagonal element m22
	mat.push_back(abs(m01Comp[0]));		//real component m01
	mat.push_back(abs(m01Comp[1]));		//imag component m01
	mat.push_back(abs(m02Comp[0]));		//real component m02
	mat.push_back(abs(m02Comp[1]));		//imag component m02
	mat.push_back(abs(m12Comp[0]));		//real component m12
	mat.push_back(abs(m12Comp[1]));		//imag component m12

}

void ConvertToVector(vector<Mat>& coherencyMat, vector<vector<float>>& coherencyVec) {
	for (int cnt = 0; cnt < coherencyMat.size(); cnt++) {
		vector<float> vec;
		vec.assign(coherencyMat[cnt].begin <float>(), coherencyMat[cnt].end<float>());
		coherencyVec.push_back(vec);
	}
}

/*override*/
void ConvertToVector(Mat& coherencyMat, vector<unsigned char>& coherencyVec) {
		vector<unsigned char> vec;
		vec.assign(coherencyMat.begin <unsigned char>(), coherencyMat.end<unsigned char>());
		coherencyVec = vec;
}

void Feature::GetCoherencyFeatures(Data data, vector<vector<float>>& resultVec, vector<unsigned char>& classValue) {
	Mat hh, vv, hv;
	Utils util;
	int winSize = 3;
	hh = data.data[0];
	vv = data.data[1];
	hv = data.data[2];

	vector<vector<double>> pauliVec;
	vector<Mat> pauli;
	this->getPauliBasis(hh, vv, hv, pauli);
	vector<Mat> coherencyMat;	
	vector<vector<float>> coherencyVec;
	Mat labelMap;
	labelMap = Mat::zeros(data.labelImages[0].size(), CV_8UC1);
	this->GetCoherencyMat(pauli, coherencyMat, winSize);

	ConvertToVector(coherencyMat, coherencyVec);	
	copy(coherencyVec.begin(), coherencyVec.end(), back_inserter(resultVec));
	util.generateLabelMap(data.labelImages, labelMap);
	ConvertToVector(labelMap, classValue);

	for (auto& e : resultVec) {
		vector<float> temp;
		temp = this->logTransform(e);

		e = temp;
	}

}


// texture feature vector length: 64
/***************************************************************
Author  : Jun Xiang with modifications by Anupama Rajkumar
Date	: 11.06.2020
****************************************************************/
void Feature::GetTextureFeature(vector<Mat>& features, vector<unsigned char>& classValue, Data data, bool flag) {

	featureName = "Texture";
	int cnt = 0;
	cout << "Starting to calculate texture features......" << endl;
	int pStart_r, pStart_c, pEnd_r, pEnd_c;
	vector <Point2i> samples;
	if (flag == true) {
		samples = data.trainSamples.Samples;
	}
	else {
		samples = data.testSamples.Samples;
	}
 
	for (const auto& p : samples) {
		
		pStart_r = int(p.x) - data.sizeOfPatch / 2;
		pStart_c = int(p.y) - data.sizeOfPatch / 2;
		Rect roi = Rect(pStart_c, pStart_r, data.sizeOfPatch, data.sizeOfPatch);
		
		if (roi.x >= 0 && roi.y >= 0 && roi.width + roi.x < data.labelImages[0].cols && roi.height + roi.y < data.labelImages[0].rows) {
			
			vector<Mat> temp;
			// intensity of HH channel
			Mat hh = logTransform(getComplexAmpl(data.data[0](roi)));
			// intensity of VV channel
			Mat vv = logTransform(getComplexAmpl(data.data[1](roi)));
			// intensity of HV channel
			Mat hv = logTransform(getComplexAmpl(data.data[2](roi)));

			temp.push_back(hh);
			temp.push_back(vv);
			temp.push_back(hv);
			Mat result;
			for (const auto& t : temp) {
				hconcat(cvFeatures::GetGLCM(t, 8, GrayLevel::GRAY_8, 32), cvFeatures::GetLBP(t, 1, 8, 32), result);
				features.push_back(result);
				if (flag == true) {
					classValue.push_back(data.trainSamples.labelName[cnt]);
				}
				else {
					classValue.push_back(data.testSamples.labelName[cnt]);
				}
				
			}
			cnt++;
			if ((cnt + 1) % 1000 == 0) {
				cout << cnt << "samples converted out of" << data.trainSamples.Samples.size() << endl;
			}
		}
		else {
			cout << "roi.x, roi.y, roi.width + roi.x, roi.height + roi.y: " << roi.x << " " << roi.y << " " << roi.width + roi.x << " " <<roi.height + roi.y <<  endl;
			cout << "rows x cols: " << data.labelImages[0].rows << "x" << data.labelImages[0].cols;
			cerr << "Roi index is not ok for " << p.x << " and " << p.y;
			exit(-1);
		}
	}
	cout << "Texture feature calculation over!!!" << endl;;
}


//#endif

