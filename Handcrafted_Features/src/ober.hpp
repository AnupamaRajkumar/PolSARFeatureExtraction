#pragma once
#ifndef  OBER_HPP_
#define  OBER_HPP_
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <string>
#include <iostream>
#include <fstream>
#include <random>
#include "specklefilter.hpp"
#include "cvFeatures.hpp"
#include "sarFeatures.hpp"
#include "dataset_hdf5.hpp"
#include "Utils.h"

/*defining compiler versions for some
compiler specific includes*/
#define VC				//GCC/VC



//load data from Oberpfaffenhofen
class ober{

private:
	unsigned border = 3;

	std::vector<cv::Point> samplePoints;
	std::vector<unsigned char> sampleLabel;

public:
	// data = complex mat with values [HH, VV, HV]
	  std::vector<cv::Mat> data;

	  cv::Mat LabelMap;

	// record the class name of each label
	std::map<unsigned char, std::string>classNames; 

	// constructor
	// input: rat file folder, label file folder 
	ober(const std::string& RATfileFolder, const std::string& labelFolder) {

		samplePoints = std::vector<cv::Point>();
		sampleLabel = std::vector<unsigned char>();

		// read rat data, can't save them directly to hdf5, it will lost precision
		loadData(RATfileFolder);

		// read labels
		std::vector<cv::Mat> masks;
		std::vector<std::string>  labelNames;
		ReadClassLabels(labelFolder, labelNames, masks);
		this->LabelMap = Utils::generateLabelMap(masks);

		// read the labelNames to dict
		classNames[unsigned char(0)] = "Unclassified";
		std::cout << "Unclassified" << " label : " << std::to_string(0) << std::endl;
		for (int i = 0; i < labelNames.size(); i++) {
			classNames.insert(std::pair<unsigned char, std::string>(i + 1, labelNames[i]));
			std::cout << labelNames[i] << " label : " << std::to_string(i+1) << std::endl;
		}
	}

	// constructor
	ober(const std::vector<cv::Mat>& sardata, const cv::Mat& labelmap, const std::map<unsigned char, std::string>& classnames) {
		this->data = sardata;
		this->LabelMap = labelmap;
		this->classNames = classnames;

	}
	~ober() {
		
	}

	

	//shuffle the samples and split them into batches with proper class distribution
	//calulate features and save to hdf5 file
	void caculFeatures(const std::string& hdf5_fileName, const std::string& feature_name, int filterSize, int patchSize, int batchSize = 5000, int numOfSamplePoint =0, unsigned char classlabel =255);
	
private:
	// write labelmap and classNames save to hdf5
	void writeLabelMapToHDF(const std::string& hdf5_fileName, cv::Mat& labelMap, std::map<unsigned char, std::string>& classNames);

	void LoadSamplePoints(const int& sampleSize, const int& numOfSamplePoint, const unsigned char& classlabel, int stride = 1);

	void getSample(const cv::Point& p, int patchSize, int filtersize, cv::Mat& hh, cv::Mat& vv, cv::Mat& hv);
	
	void getSampleInfo(const std::string& hdf5_fileName, const cv::Mat& pts, int patchSize);

	// get texture features(LBP and GLCM) on HH,VV,VH
	cv::Mat caculTexture(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv);

	// get color features(MPEG-7 DCD,CSD) on Pauli Color image
	cv::Mat caculColor(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv);

	// get MP features on HH,VV,VH 
	cv::Mat caculMP(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv);

	// get polsar features on elements of covariance matrix C and coherency matrix T
	cv::Mat caculCTelements(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv);

	// get polsar features on target decompostion 
	cv::Mat caculDecomp(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv);

	// get polsar features on statistic of polsar parameters
	cv::Mat caculPolStatistic(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv);

	

	/***Author: Anupama Rajkumar***/
	void loadData(std::string RATfolderPath);
	void ReadClassLabels(std::string labelPath, std::vector<std::string>& labelNames, std::vector<cv::Mat>& labelImages);
	cv::Size loadRAT(std::string fname, std::vector<cv::Mat>& data, bool metaOnly = false);
	cv::Size loadRAT2(std::string fname, std::vector<cv::Mat>& data, bool metaOnly = false);
	void getTileInfo(cv::Size size, unsigned border, unsigned& tile, std::vector<unsigned>& tileSize, std::vector<unsigned>& tileStart);
};

#endif