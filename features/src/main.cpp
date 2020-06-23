//#include "torchDataSet.cpp"
//#include "sen12ms.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "ober.hpp"
#include "Utils.h"
#include "sarFeatures.hpp"
#include "cvFeatures.hpp"
#include "Param.h"
#include "KNN.hpp"
#include <iostream>
#include <fstream>


using namespace std;
using namespace cv;
 
int main() {

    string ratfolder = "E:\\Oberpfaffenhofen\\sar-data";
    string labelfolder = "E:\\Oberpfaffenhofen\\label";
    string oberfile = "E:\\ober.h5";

    ober* ob = new ober(ratfolder, labelfolder, oberfile);
    KNN* knn = new KNN();
    int k = 20;

    // set patch size 10, maximum sample points per class is 100
    int patchSize = 20;
    int patchNumPerClass = 50;
    int filterSize = 5;

    ob->LoadSamplePoints(patchSize, patchNumPerClass);
    //// or load sample points from outside
    // vector<Point> samplePoints;
    // vector<unsigned char> pointLabels;
    // ob->LoadSamplePoints(samplePoints, pointLabels, patchSize);

    ob->setFilterSize(filterSize);

    vector<string> feature_type = { texture, color,MP,ctElements,decomp,statistic };
    for (size_t i = 0; i < feature_type.size(); i++) {
        vector<Mat> features;
        vector<unsigned char> labels;
        cout << endl;
        cout << "calculate " << feature_type[i] << " with filterSize " << filterSize << " , patchSize " << patchSize<< endl;
        switch (i) {
        case 0:
            ob->GetTextureFeature(features, labels);
            break;
        case 1:
            ob->GetColorFeature(features, labels);
            break;
        case 2:
            ob->GetMPFeature(features, labels);
            break;
        case 3:
            ob->GetCTFeatures(features, labels);
            break;
        case 4:
            ob->GetDecompFeatures(features, labels);
            break;
        case 5:
            ob->GetPolsarStatistic(features, labels);
            break;
        }
        knn->applyKNN(features, labels, k, 80);
        ob->saveFeaturesToHDF(oberfile, feature_type[i], dataset_name, features, labels, filterSize, patchSize);
        features.clear();
        labels.clear();
    }

	//calculate and load all the sample points in the image


    for (size_t i = 0; i < feature_type.size(); i++) {
        vector<Mat> features;
        vector<unsigned char> labels;
        cout << endl;
        cout << "get " << feature_type[i] << " from hdf5 file with filterSize " << filterSize << " , patchSize " << patchSize << endl;
        ob->getFeaturesFromHDF(oberfile, feature_type[i], dataset_name, features, labels, filterSize, patchSize);
        knn->applyKNN(features, labels, k, 80);
        features.clear();
        labels.clear();
    }

    return 0;
}
   

