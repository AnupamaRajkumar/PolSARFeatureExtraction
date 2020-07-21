#include "Performance.h"
#include <iostream>
#include <opencv2/opencv.hpp>


double Performance::calculatePredictionAccuracy(vector<unsigned char>& classResult, vector<unsigned char>& testLabels, 
												vector<pair<vector<Point2i>, uint>>& patchPoint)
{
	double accuracy = 0.0;
	if (classResult.size() != testLabels.size()) {
		cerr << "Predicted and actual label vectors differ in length. Something doesn't seem right." << endl;
		exit(-1);
	}
	else {
		int dim = classResult.size();
		double hit, miss, city, field, forest, grassland, street;
		hit = 0;
		miss = 0;
		city = 0, field = 0, forest = 0, grassland = 0, street = 0;
		for (int i = 0; i < dim; i++) {
			if (classResult[i] == testLabels[i]) {
				//hit for true negative and true positive
				hit++;
				switch (classResult[i])
				{
				case 1:
					city++;
					break;
				case 2:
					field++;
					break;
				case 3:
					forest++;
					break;
				case 4:
					grassland++;
					break;
				case 5:
					street++;
					break;
				default:
					break;
				}
			}
			else {
				//miss for false positive and false negative
				miss++;
			}
		}
		accuracy = double(hit / dim);

		cout << "Number of city points found : " << city << ", Accuracy of city data points : " << double(city / (0.2 *patchPoint[0].first.size())) << endl;
		cout << "Number of field points found : " << field << ", Accuracy of field data points : " << double(field / (0.2 *patchPoint[1].first.size())) << endl;
		cout <<  "Number of forest points found : " << forest << ", Accuracy of forest data points : " << double(forest / (0.2 *patchPoint[2].first.size())) << endl;
		cout << "Number of grassland points found : " << grassland << ", Accuracy of grassland data points : " << double(grassland / (0.2 *patchPoint[3].first.size())) << endl;
		cout << "Number of street points found : " << street << ", Accuracy of street data points : " << double(street / (0.2 *patchPoint[4].first.size())) << endl;

	}
	return accuracy;
}

double Performance::calculatePredictionAccuracy(vector<unsigned char>& classResult, vector<unsigned char>& testLabels)
{
	double accuracy = 0.0;
	if (classResult.size() != testLabels.size()) {
		cerr << "Predicted and actual label vectors differ in length. Something doesn't seem right." << endl;
		exit(-1);
	}
	else {
		int dim = classResult.size();
		double hit, miss, city, field, forest, grassland, street;
		double totCity, totField, totForest, totGrass, totStreet;
		hit = 0;
		miss = 0;
		city = 0, field = 0, forest = 0, grassland = 0, street = 0;
		totCity = 0, totField = 0, totForest = 0, totGrass = 0, totStreet = 0;
		for (int cnt = 0; cnt < testLabels.size(); cnt++) {
			if (testLabels[cnt] == 1) {
				totCity++;
			}
			else if(testLabels[cnt] == 2) {
				totField++;
			}
			else if (testLabels[cnt] == 3) {
				totForest++;
			}
			else if (testLabels[cnt] == 4) {
				totGrass++;
			}
			else if (testLabels[cnt] == 5) {
				totStreet++;
			}
			else {
				/*do nothing*/
			}
		}
		for (int i = 0; i < dim; i++) {
			if (classResult[i] == testLabels[i]) {
				//hit for true negative and true positive
				hit++;
				switch (classResult[i])
				{
				case 1:
					city++;
					break;
				case 2:
					field++;
					break;
				case 3:
					forest++;
					break;
				case 4:
					grassland++;
					break;
				case 5:
					street++;
					break;
				default:
					break;
				}
			}
			else {
				//miss for false positive and false negative
				miss++;
			}
		}
		accuracy = double(hit / dim);
		double accCity, accField, accForest, accGrass, accStreet;
		accCity = accField = accForest = accGrass = accStreet = 0.;
		if (totCity > 0) {
			accCity = double(city / totCity);
		}
		if (totField > 0) {
			accField = double(field / totField);
		}
		if (totForest > 0) {
			accForest = double(forest / totForest);
		}
		if (totGrass > 0) {
			accGrass = double(grassland / totGrass);
		}
		if (totStreet > 0) {
			accStreet = double(street / totStreet);
		}
		cout << "Number of city points found : " << city << " from " << totCity << " points, Accuracy of city data points : " << accCity << endl;
		cout << "Number of field points found : " << field << " from " << totField << " points, Accuracy of field data points : " << accField << endl;
		cout << "Number of forest points found : " << forest << " from " << totForest << " points, Accuracy of forest data points : " << accForest << endl;
		cout << "Number of grassland points found : " << grassland << " from " << totGrass << "points, Accuracy of grassland data points : " << accGrass << endl;
		cout << "Number of street points found : " << street << " from " << totStreet << " points, Accuracy of street data points : " << accStreet << endl;
	}
	return accuracy;
}