#pragma once

/*defining compiler versions for some 
compiler specific includes*/
#define VC				//GCC/VC


#ifndef DATA_H
#define DATA_H

#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define NUMOFCLASSES  5
#define MAXNUMOFPATCHES 1

class Data {
public:
	/*constructors and desctructors*/
	 Data(void);
	~Data(void);

	/***********Functions***************/
	void loadData(string folder);
	Mat loadImage(string fname);
	void loadLabels(const string &folderPath, vector<string>& labelNames, vector<Mat>& labelImages,
					vector<vector<Point2i>>& numOfPoints);
	void loadPolSARData(std::vector<std::string> const& fname);
	void getTileInfo(cv::Size size, unsigned border, unsigned &tile, vector<unsigned> &tileSize, vector<unsigned> &tileStart);
	void ReadClassLabels(string labelPath, vector<string> &labelNames, vector<Mat> &labelImages);
	Size loadRAT(string fname, vector<Mat> &data, bool metaOnly = false);
	Size loadRAT2(string fname, vector<Mat> &data, bool metaOnly = false);
	vector<Mat> GetPatches(Mat origImg, vector<Point2i> samplePoints, int sizeOfPatch);
	/***********Functions***************/

	/***********Variables***************/
	// data = scattering vector with values [HH, VV, HV]
	std::vector<cv::Mat> data;
	unsigned border = 3;
	int sizeOfPatch = 10;
	vector<Mat> labelImages;
	vector<string> labelNames;
	vector<vector<Point2i>> numOfPoints;

	typedef struct {
		vector<Point2i> Samples;
		vector<unsigned char> labelName;
	} trainTestData;

	trainTestData trainSamples;
	trainTestData testSamples;

	/***********Variables***************/

};


#endif