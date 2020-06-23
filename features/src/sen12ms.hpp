#ifndef SEN12MS_HPP_
#define SEN12MS_HPP_

#include <opencv2/opencv.hpp>
#include "Geotiff.hpp"
#include "Utils.h"
#include <string>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

// two types of masks
// choose in (IGBP, LCCS)
enum class MaskType
{
	IGBP = 0,
	LCCS = 1
};



// to load sen12ms dataset
class sen12ms {
public:
	 vector<std::string>  s1FileList;
	 vector<std::string>  lcFileList;
	

private:
	 vector<Mat>* list_images;  // false color images(unnormalized)
	 vector<Mat>* list_labelMaps; // label map
	 
	 int batchSize;
	 MaskType mask_type;

	 // draw samples from each image of mask area
	 int sampleSize = 10; 
	 // Maximum sample points of each mask area
	 int samplePointNum = 100;

public:
	// Constructor
	sen12ms(const string& s1FileListPath, const string & lcFileListPath) {
		loadFileList(s1FileListPath, lcFileListPath);
		list_images = new vector<Mat>();
		list_labelMaps = new vector<Mat>();
		batchSize = 0;
		mask_type = MaskType::IGBP;
	}

	sen12ms(vector<std::string>  &s1List, vector<std::string> & lcList) {
		s1FileList = s1List;
		lcFileList = lcList;
		list_images = new vector<Mat>();
		list_labelMaps = new vector<Mat>();
		batchSize = 0;
		mask_type = MaskType::IGBP;
	}

	~sen12ms()
	{
		if(list_images){ delete list_images; }
		if(list_labelMaps) { delete list_labelMaps; }
		 
	}

	
	void SetMaskType(MaskType maskType) {
		mask_type = maskType;
	}

	void SetBatchSize(int size) {
		batchSize = size;

		if (!list_images) free(list_images); 
		if (!list_labelMaps) free(list_labelMaps);
		 

		list_images = new vector<Mat>(batchSize);
		list_labelMaps = new vector<Mat>(batchSize);
	}

	// set the sample size and Maximum sample points of each mask area
	void SetSample(const int &size,const int & num) {
		sampleSize = size;
		samplePointNum = num;
	}

	void GetData(vector<Mat>& images, vector<Mat>& labelMaps) {
		images = *list_images;
		labelMaps = *list_labelMaps;
	}

	// load current batch to memory 
	void LoadBatchToMemeory(int batch);

	// be careful to use this function
	void LoadAllToMemory();
	
	// get training data
	void GetPatches(vector<Mat>& patches, vector<unsigned char>& classValue);

	// Get PNG files for images and maskes
	void GeneratePNG(const string& outputpath);

	string GetClassName(signed char classValue);
	

private:
	// Load tiff file list
	void loadFileList(const string& s1FileListPath, const string& lcFileListPath);

	// Merge LCCS_LC, LCCS_LU,LCCS_SH into LCCS class
	//Generate IGBP, LCCS from the ground truth 
	void getLabelMap(const Mat& lc, Mat& labelMap);
	
	//check if cetain class type existed in a class category
	bool findLandClass(const Mat& labelMap, vector<std::pair<int, int> >& ind, const unsigned char& landclass);

	// Create Masks for each patch
	void getMask(const Mat& labelMap, vector<Mat>& list_masks, vector<unsigned char>& list_classValue);

	// Generate false color image from SAR data
	// R: VV, G:VH, B: VV/VH
	Mat getFalseColorImage(const Mat& src, bool normed);

	// Generate samples from each img
	void getSamples(const Mat& img, const Mat& mask, const unsigned char& mask_label, vector<Mat>& samples, vector<unsigned char>& sample_labels);
};

#endif
 