#include "Visualisation.h"

/***********************************************************************
Calculate the mean of each feature vector in the feature map
Author : Anupama Rajkumar
Date : 21.07.2020
Description: This function calculates the mean of each feature point if the
size of the feature vector > 1
Input 1 : The calculated feature vector
Input 2 : The calculated mean output which is also linearly stretched for
contrast correction
*************************************************************************/
void Visual::CalculateMeanFeatureVector(vector<vector<float>>& featureVector, Mat& outPut) {

		vector<float> outVec, outVecCpy;
		for (int cnt = 0; cnt < featureVector.size(); cnt++) {
			float mean = 0.;
			for (int s = 0; s < featureVector[cnt].size(); s++) {
				mean += featureVector[cnt][s];
				mean /= featureVector[cnt].size();
			}
			outVec.push_back(mean);
		}
		outVecCpy = outVec;
		float min = 0., max = 0.;
		/*calculate min and max value of features for contrast correction*/
		sort(outVecCpy.begin(), outVecCpy.end());
		min = outVecCpy[0];
		max = outVecCpy[outVec.size() - 1];
		cout << "min:" << min << ", max:" << max << endl;
		/*convert vector to Mat*/
		Mat OutMat;
		OutMat = Mat::zeros(outPut.rows, outPut.cols, CV_32FC1);
		if (outVec.size() == outPut.rows * outPut.cols) {
			memcpy(OutMat.data, outVec.data(), outVec.size() * sizeof(float));
		}


		/*scale values of outMat to outPut*/
		for (int row = 0; row < OutMat.rows; row++) {
			for (int col = 0; col < OutMat.cols; col++) {
				/*linear gray scale contrast stretching*/
				float val = ((OutMat.at<float>(row, col) - min)* 255.0) / (max - min);
				outPut.at<float>(row, col) = val;
			}
		}
}

/***********************************************************************
Generating the feature map
Author : Anupama Rajkumar
Date : 21.07.2020
Description: This function generates the visualisation of the calculated
feature map
Input : The calculated feature vector
*************************************************************************/


void Visual::GenerateFeatureMap(vector<vector<float>>& m_featureVector) {
	/*Visualising the feature map*/
	cout << "Visualising the feature vector...." << endl;
	Mat meanMat;
	int row = 6640, col = 1390;

	meanMat = Mat(row, col, CV_32FC1);
	this->CalculateMeanFeatureVector(m_featureVector, meanMat);
	/*Generate colormap*/
	cout << "Generating colormap" << endl;
	Mat outMat;
	meanMat.convertTo(meanMat, CV_8UC1);
	outMat = meanMat.clone();
	/*apply color map*/
	cv::Mat mlookUpTable_8UC1(1, 256, CV_8UC1);
	for (int i = 0; i < 256; ++i)
	{
		mlookUpTable_8UC1.at<uchar>(0, i) = uchar(255 - i);
	}
	cv::LUT(meanMat, mlookUpTable_8UC1, outMat);
	/*equalising histogram for better contrast*/
	cv::equalizeHist(outMat, outMat);
	cv::imwrite("PatchColorMap_AE.png", outMat);
}