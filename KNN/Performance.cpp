#include "Performance.h"
#include <iostream>
#include <opencv2/opencv.hpp>


double Performance::calculatePredictionAccuracy(vector<string>& classResult, vector<string>& testLabels)
{
	double accuracy = 0.0;
	if (classResult.size() != testLabels.size()) {
		cerr << "Predicted and actual label vectors differ in length. Somethig doesn't seem right." << endl;
		exit(-1);
	}
	else {
		int dim = classResult.size();
		double hit, miss;
		hit = 0;
		miss = 0;
		for (int i = 0; i < dim; i++) {
			if (classResult[i] == testLabels[i]) {
				hit++;
			}
			else {
				miss++;
			}
		}
		accuracy = double(hit / dim);
	}
	return accuracy;
}