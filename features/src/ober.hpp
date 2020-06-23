#pragma once
#ifndef  OBER_HPP_
#define  OBER_HPP_
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "specklefilter.hpp"
#include "sarFeatures.hpp"
#include "Utils.h"

/*defining compiler versions for some
compiler specific includes*/
#define VC				//GCC/VC

using namespace std;
using namespace cv;

//load data from Oberpfaffenhofen
class ober{

private:
	unsigned border = 3;

	vector<Point> samplePoints;
	vector<unsigned char> samplePointClassLabel;
	int sampleSize;
	int filterSize;
	string hdf5_file;

public:
	// data = complex mat with values [HH, VV, HV]
	  vector<Mat> data;
	  vector<unsigned char> maskLabels;  
	  vector<Mat> masks;

	// record the class name of each label
	std::map<unsigned char, string>classNames; 

	// constructor
	// input: rat file folder, label file folder 
	ober(const string& RATfileFolder, const string& labelFolder, const string & hdf5_fileName) {

		samplePoints = vector<Point>();
		samplePointClassLabel = vector<unsigned char>();
		sampleSize = 0;
		 
		hdf5_file = hdf5_fileName;

			// read rat data, can't save them directly to hdf5, it will lost precision
			loadData(RATfileFolder);
			if( data.size() ==3){
				data[0].convertTo(data[0], CV_64FC2);
				data[1].convertTo(data[1], CV_64FC2);
				data[2].convertTo(data[2], CV_64FC2);
			}
			else if (data.size() == 2) {
				data[0].convertTo(data[0], CV_64FC2);
				data[1].convertTo(data[1], CV_64FC2);
			}

		// read labels
		vector<string>  labelNames;
		ReadClassLabels(labelFolder, labelNames, masks);

		// read the labelNames to dict
		classNames[signed char(0)] = "Unclassified";
		for (int i = 0; i < labelNames.size(); i++) {
			classNames.insert(pair<unsigned char, string>(i + 1, labelNames[i]));
			maskLabels.push_back(i + 1);
		}

		writeLabelMapToHDF(hdf5_file);
	}


	~ober() {
		
	}

	// apply refined Lee despeckling filter, choose from ( 5, 7, 9, 11)
	// void applyRefinedLeeFilter( int filter_size);
	void setFilterSize(int size) { this->filterSize = size; }

	// input sample size and the maximum number of sample points per class 
	 void LoadSamplePoints(const int& sampleSize, const int& samplePointNum);
	 void LoadSamplePoints(const vector<Point>& samplePoints, const vector<unsigned char>& pointLabels, const int& sampleSize);

	 // get patches of 3 channel (HH+VV,HV,HH-VV) intensity(dB)
	 void GetPauliColorPatches(vector<Mat>& patches, vector<unsigned char>& classValue);

	 // get patches of 3 channel (HH,HV,VV) intensity(dB)
	 void GetPatches(vector<Mat>& patches, vector<unsigned char>& classValue);

	 // get texture features(LBP and GLCM) on HH,VV,VH
	 void GetTextureFeature(vector<Mat>& features, vector<unsigned char>& classValue);

	 // get color features(MPEG-7 DCD,CSD) on Pauli Color image
	 void GetColorFeature(vector<Mat>& features, vector<unsigned char>& classValue);

	 // get MP features on HH,VV,VH 
	 void GetMPFeature(vector<Mat>& features, vector<unsigned char>& classValue);
	 
	 // calculate covariance and coherency matrix and store to hdf5 file
	 // get polsar features on elements of covariance matrix C and coherency matrix T
	 void GetCTFeatures(vector<Mat>& features, vector<unsigned char>& classValue);

	 // get polsar features on target decompostion 
	 void GetDecompFeatures(vector<Mat>& features, vector<unsigned char>& classValue);

	 // get polsar features on statistic of polsar parameters
	 void GetPolsarStatistic(vector<Mat>& features, vector<unsigned char>& classValue);

	 void getFeaturesFromHDF(const String& hdf5_fileName, const String& parent_name, const vector<String>& dataset_name, 
		 vector<Mat>& features, vector<unsigned char>& featureLabels, int filterSize =5, int patchSize =20);
	 
	 // write calculated features to hdf5 ( sample points, labels, features)
	 void saveFeaturesToHDF(const String& hdf5_fileName, const String& parent_name, const vector<String>& dataset_name, 
		 vector<Mat>& features,  vector<unsigned char>& featureLabels, int filterSize =5, int patchSize =20);


	 
	 


private:
	// calculate target decompostion features
	// vector<mat> result, vector length: , mat size: (hh.rows,hh.cols)
	void getTargetDecomposition(const Mat & hh, const Mat &vv, const Mat& hv, vector<Mat>& result);
	
	// apply refined Lee filter to samples, filterSize choose from (5,7,9,11)
	void getSample(const Point& sample_point, int patchSize, int filtersize, Mat& hh, Mat& vv, Mat& hv);

	void getSamplePointInfo(const string& hdf5_fileName, const Mat& pts, int patchSize);
	void writeLabelMapToHDF(const string& hdf5_fileName);
	void generateSamplePoints(const string& hdf5_fileName,int patchSize);
	void getSamplePoints(const string& hdf5_fileName, vector<Point>& pts, vector<unsigned char>& pointLabels, int patchSize, int numOfSamplePoints);

	/***Author: Anupama Rajkumar***/
	void loadData(string RATfolderPath);
	void ReadClassLabels(string labelPath, vector<string>& labelNames, vector<Mat>& labelImages);
	Size loadRAT(string fname, vector<Mat>& data, bool metaOnly = false);
	Size loadRAT2(string fname, vector<Mat>& data, bool metaOnly = false);
	void getTileInfo(cv::Size size, unsigned border, unsigned& tile, vector<unsigned>& tileSize, vector<unsigned>& tileStart);
};

#endif