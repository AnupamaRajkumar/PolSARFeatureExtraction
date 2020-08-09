#pragma once
#ifndef UTILS_H
#define UTILS_H


#include "DataProcess.hpp"
#include "sarFeatures.hpp"
#include "dataset_hdf5.hpp"


namespace Utils {

		//Read the features from hdf5 file, classify them and write the classifiy results into hdf5 file
		void classifyFeaturesML(const std::string& hdf5_fileName, const std::string& feature_name, const std::string classifier_type, int trainPercent, int filterSize, int patchSize, int batchSize);

		//Generate the colormap of classified results
		void generateColorMap(const std::string& hdf5_fileName, const std::string& feature_name, const std::string& classifier_type, int filterSize,int patchSize, int batchSize);
	    
		//Get the visulization of feature map for single feature
		void generateFeatureMap(const std::string& hdf5_fileName, const std::string& feature_name, int filterSize, int patchSize, int batchSize);

		cv::Vec3b getLabelColor(unsigned char class_label);
	    
		//reduced the feature dimension by T-SNE, dump to txt for plotting
		void featureDimReduction(const std::string& hdf5_fileName, const std::string& feature_name, int numSamples, int filterSize, int patchSize);

		//get features data and its groundtruth from hdf5
		void getFeaturesFromHDF(const std::string& hdf5_fileName, const std::string& feature_name, 
			std::vector<cv::Mat>& features, std::vector<unsigned char>& featureLabels, std::vector<cv::Point>& labelPoints,  int offset_row = 0, int counts_rows = 0);
		
		void saveClassResultToHDF(const std::string& hdf5_fileName, const std::string& feature_name, const std::string& classResult_name,
			const std::vector<unsigned char>& class_result, const std::vector<cv::Point>& points);
		
		//generate all the possible sample points
		std::vector<cv::Point>  generateSamplePoints(const cv::Mat& labelMap, const int& sampleSize, const int & stride );
	 
		//get random samples of homogeneous area for one type of class
		void getRandomSamplePoint(const cv::Mat& labelMap, std::vector<cv::Point>& samplePoints, const unsigned char& sampleLabel, const int& sampleSize, const int& stride, const int& numOfSamplePointPerClass);
		
		cv::Mat generateLabelMap(const std::vector<cv::Mat>& masks);
		
		//split index to batches, make sure the distribution of each class in each batch is the same as it in the whole data
		void splitVec(const std::vector<unsigned char>& labels, std::vector<std::vector<int>>& subInd, int batchSize=5000);

		std::map<unsigned char, std::string> getClassName(const std::string& filename);

};
#endif
