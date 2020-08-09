#include "Visualisation.h"

/***********************************************************************
Perform contrast correction on each feature vector in the feature map
Author : Anupama Rajkumar
Description: This functions converts vector to matrix and performs
contrast correction by applying gray scale stretching and histogram 
equalisation
Input 1 : The calculated feature vector
Input 2 : The count number of the current feature vector
Output	: Contast corrected matrix
*************************************************************************/
void Visual::ContrastCorrection(vector<vector<float>>& featureVector, int cnt, Mat& outPut) {

		vector<float> outVec, outVecCpy;
		outVec.clear();
		outVecCpy.clear();
		for (int s = 0; s < featureVector.size(); s++) {
			outVec.push_back(featureVector[s][cnt]);
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
Description: This function generates the visualisation of the calculated
feature map after contrast correction
Input 1 : The calculated feature vector
Input 2 : Generated colormap image name
*************************************************************************/

void Visual::GenerateFeatureMap(vector<vector<float>>& m_featureVector, string& imagePrefix) {
	/*Visualising the feature map*/
	cout << "Visualising the feature vector...." << endl;
	
	int row = 6640, col = 1390;		
	int size = m_featureVector[0].size();
	for (int cnt = 0; cnt < size; cnt++) {
		cout << "Feature map " << cnt + 1 << endl;
		Mat featureMat;
		featureMat = Mat(row, col, CV_32FC1);
		this->ContrastCorrection(m_featureVector, cnt, featureMat);
		/*Generate colormap*/
		cout << "Generating colormap" << endl;
		Mat outMat;
		featureMat.convertTo(featureMat, CV_8UC1);
		outMat = featureMat.clone();
		/*equalising histogram for better contrast*/
		cv::equalizeHist(outMat, outMat);
		cv::applyColorMap(outMat, outMat, COLORMAP_JET);
		string filename = imagePrefix + to_string(cnt) + ".png";
		cv::imwrite(filename, outMat);
	}

}