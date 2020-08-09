#pragma once
#include "cv_hdf5.hpp"
#include <opencv2/opencv.hpp>

#ifndef DATASET_HDF5_H
#define DATASET_HDF5_H

namespace hdf5 {

	//************* HDF5 file read/write/insert/delete *****************//
	bool checkExist(const std::string& filename, const std::string& parent_name, const std::string& dataset_name);

	bool insertData(const std::string& filename, const std::string& parent_name, const std::string& dataset_name, const cv::Mat& data);

	void readData(const std::string& filename, const std::string& parent_name, const std::string& dataset_name, cv::Mat& data, int offset_row = 0, int counts_rows = 0);
	
	void deleteData(const std::string& filename, const std::string& parent_name, const std::string& dataset_name);
	
	void writeData(const std::string& filename, const std::string& parent_name, const std::string& dataset_name, const cv::Mat& data);
	
	int getRowSize(const std::string& filename, const std::string& parent_name, const std::string& dataset_name);

	int getColSize(const std::string& filename, const std::string& parent_name, const std::string& dataset_name);
	
	//write attribute to the root group
	void writeAttr(const std::string& filename, const std::string& attribute_name, const int& attribute_value);

	void writeAttr(const std::string& filename, const std::string& attribute_name, const cv::Mat& attribute_value);
	
	void writeAttr(const std::string& filename, const std::string& attribute_name, const std::string& attribute_value);
	
	void readAttr(const std::string& filename, const std::string& attribute_name, int& attribute_value);
	
	void readAttr(const std::string& filename, const std::string& attribute_name, std::string& attribute_value);

	void readAttr(const std::string& filename, const std::string& attribute_name, cv::Mat& attribute_value);
};


#endif