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

class Data {
public:
	/*constructors and desctructors*/
	 Data(void);
	~Data(void);

	/***********Functions***************/
	void loadData(string folder);
	Mat loadImage(string fname);
	void loadLabels(const string &folderPath, vector<Mat>& labelImages, vector<string>& labelNames, vector<vector<Point2i>>& numOfPoints);
	void loadPolSARData(std::vector<std::string> const& fname);
	void getTileInfo(cv::Size size, unsigned border, unsigned &tile, vector<unsigned> &tileSize, vector<unsigned> &tileStart);
	Size loadRAT(string fname, vector<Mat> &data, bool metaOnly = false);
	Size loadRAT2(string fname, vector<Mat> &data, bool metaOnly = false);
	void ExtractLabelPatches(vector<vector<Point2i>> numOfPoints, int numOfSamples, int sizeOfPatch, vector<Mat> labelImages, vector<vector<Point2i>>& samplePoints);
	void ExtractImagePatches(int numOfSamples, int sizeOfPatch, Mat RGBImg, vector<vector<Point2i>> samplePoints, vector<vector<Mat>>& patches);
	void ExtractImagePoints(int numOfSamples, Mat& RGBImg, vector<Point2i>& samples);
	vector<Point2i> DrawSamples(vector<Point2i> numOfPoints, int numOfSamples, int sizeOfPatch, Mat labelImages);
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
		vector<string> labelName;
	} trainTestData;

	trainTestData trainSamples;
	trainTestData testSamples;

	/***********Variables***************/

};


#endif