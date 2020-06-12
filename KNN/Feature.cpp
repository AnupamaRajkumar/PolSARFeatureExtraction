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

// texture feature vector length: 64
/***************************************************************
Author  : Jun Xiang with modifications by Anupama Rajkumar
Date	: 11.06.2020
****************************************************************/
void Feature::GetTextureFeature(vector<Mat>& features, vector<string>& classValue, Data data, bool flag) {

	Utils utils;
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

