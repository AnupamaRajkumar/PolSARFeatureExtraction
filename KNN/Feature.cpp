/**************************************************
Step 1: Form RGB Image (Eg use Pauli decomposition)
Step 2: Oversegmentation
Step 3: Extract feature vectors based on RGB Image 
Step 4: Generate a new probablistic metric by applying softmax
Step 5: Introduce this to KNN to improve accuracy of classification

****************************************************/

#include "Feature.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

//using namespace std;
//using namespace cv;

#if 0
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

Mat Feature::RGBFeaturesFromPauli(vector<Mat>& lexi, vector<Mat>& pauli) {
	Mat RGBImage;
	lexi2pauli(lexi, pauli);
	/*calculate abs(alpha)^2, abs(beta)^2, abs(gamma)^2*/
	/*save in the RGB Image*/
	/*use patches from that RGB image for testing*/
}

#endif