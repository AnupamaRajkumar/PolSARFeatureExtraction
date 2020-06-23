#include "Utils.h"
#include <iostream>
#include <fstream>
#include <random>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include "sarFeatures.hpp"
#include "cv_hdf5.hpp"


using namespace std;
using namespace cv;

/**********************************
Generating a label map
Author : Anupama Rajkumar
Date : 27.05.2020
Modified by: Jun Xiang 22,06,2020
Description : Idea is to create a single label map from a list of various
label classes.This map serves as points of reference when trying to classify
patches
* ************************************************************************/

Mat Utils::generateLabelMap(const vector<Mat> & masks) {
	size_t NUMOFCLASSES = masks.size();
	int rows = masks[0].rows;
	int cols = masks[0].cols;
	Mat labelMap = Mat::zeros(rows, cols, CV_8UC1);
	for (size_t cnt = 0; cnt < NUMOFCLASSES; cnt++) {
		Mat mask = masks[cnt];
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				if (labelMap.at<unsigned char>(row, col) == (unsigned char)(0)) {
					if (mask.at<unsigned char>(row, col) > (unsigned char)(0)) {
						labelMap.at<unsigned char>(row, col) = static_cast<unsigned char>(cnt + 1);		    //class of label
					}
				}
			}
		}
	}
	return labelMap;
}

/***********************************************************************
A helper function to store the classification data
Author : Anupama Rajkumar
Date : 27.05.2020
Description: Using this function store classification values in csv files
that can be used later for data analysis
*************************************************************************/

void Utils::WriteToFile(Mat& labelMap, string& fileName) {
	ofstream distance_list;
	distance_list.open(fileName);

	for (int row = 0; row < labelMap.rows; row++) {
		for (int col = 0; col < labelMap.cols; col++) {
			distance_list << labelMap.at<float>(row, col) << ",";
		}
		distance_list << endl;
	}
}



/***********************************************************************
A helper function containing color metadata to be used when visualizing
Author : Eli Ionescu
Date : 27.05.2020
Description: Using this function creates a map of the colors and the labels
they correspond to. To be used with visualization
*************************************************************************/
map<string, Vec3f> Utils::loadLabelsMetadata()
{
	map<string, Vec3f> name_color;

	// Color is BGR not RGB!
	Vec3f red = Vec3f(49.0f, 60.0f, 224.0f);
	Vec3f blue = Vec3f(164.0f, 85.0f, 50.0f);
	Vec3f yellow = Vec3f(0.0f, 190.0f, 246.0f);
	Vec3f dark_green = Vec3f(66.0f, 121.0f, 79.0f);
	Vec3f light_green = Vec3f(0.0f, 189.0f, 181.0f);
	Vec3f black = Vec3f(0.0f, 0.0f, 0.0f);

	name_color["city"] = red;
	name_color["field"] = yellow;
	name_color["forest"] = dark_green;
	name_color["grassland"] = light_green;
	name_color["street"] = blue;
	name_color["unclassified"] = black;

	return name_color;
}

/***********************************************************************
A helper function to visualize the maps (label or classified)
Author : Eli Ionescu
Date : 27.05.2020
Description: Using this function to assign colors to maps (label and classified)
*************************************************************************/
Mat_<Vec3f> Utils::visualiseLabels(Mat &image, string& imageName)
{
	map<string, Vec3f> colors = loadLabelsMetadata();

	Mat result = Mat(image.rows, image.cols, CV_32FC3, Scalar(255.0f, 255.0f, 255.0f));
	// Create the output result;
	for (int row = 0; row < image.rows; row++) {
		for (int col = 0; col < image.cols; col++) {
			Vec3f color;
			// Take every point and assign the right color in the result Mat
			int val = image.at<float>(row, col);
			switch (val) {
			case 0:
				color = colors["unclassified"];
				break;
			case 1:
				color = colors["city"];
				break;
			case 2:
				color = colors["field"];
				break;
			case 3:
				color = colors["forest"];
				break;
			case 4:
				color = colors["grassland"];
				break;
			case 5:
				color = colors["street"];
				break;
			default:
				cout << "Wrong value" << endl;
				break;
			}
			result.at<Vec3f>(row, col) = color;
		}
	}
	imwrite(imageName, result);

	return result;
}

/***********************************************************************
A helper function to visualize the maps (label or classified)
Author : Anupama Rajkumar
Date : 01.06.2020
Description: Using this function to create visualizations by reading the stored
csv files
*************************************************************************/

void Utils::Visualization(string& fileName, string& imageName, Size size) {

	cv::Mat img;
	//reading data from csv
	cv::Ptr<cv::ml::TrainData> raw_data = cv::ml::TrainData::loadFromCSV(fileName, 0, -1, -1);
	cv::Mat data = raw_data->getSamples();
	// optional if you have a color image and not just raw data
	data.convertTo(img, CV_32FC1);
	// set the image size
	cv::resize(img, img, size);
	//visualize the map
	visualiseLabels(img, imageName);
	cv::waitKey(0);
}

/***********************************************************************
Helper function to get the patch start and end index from label map
make sure the patch stays in one type of class areas
Author : Anupama Rajkumar
Date : 29.05.2020
Modified by: Jun Xiang
*************************************************************************/

void Utils::GetPatchIndex(int sizeOfPatch, Point2i &samplePoint,const Mat& LabelMap, int& start_col, int& start_row, int& end_col, int& end_row) {
	// (x,y)->(col,row)
	start_col = samplePoint.x - (sizeOfPatch / 2.);
	start_row = samplePoint.y - (sizeOfPatch / 2.);

	end_col = samplePoint.x + (sizeOfPatch / 2.);
	end_row = samplePoint.y + (sizeOfPatch / 2.);

	if ((start_col < 0) || (start_row < 0))
	{
		start_col = 0;
		start_row = 0;
		end_col = sizeOfPatch;
		end_row = sizeOfPatch;
	}
	if ((end_row > LabelMap.rows) || (end_col > LabelMap.cols))
	{
		start_row = LabelMap.rows - 1 - sizeOfPatch;
		start_col = LabelMap.cols - 1 - sizeOfPatch;
		end_row = LabelMap.rows - 1;
		end_col = LabelMap.cols - 1;
	}
}


/*===================================================================
 * Function: getSafeSamplePoints
 * Author: Jun Xiang
 *
 * Summary:
 *   Extract sample points from mask area or any img
 *
 * Arguments:
 *   Mat& mask  --  binarized mask
 *   const int& samplePointNum -- maximum number of sample points for mask area
 *   const int& sampleSize  -- patch size at the sample point
 *	 vector<Point>& pts  --- to record the index of the sample points
 * Returns:
 *   void
=====================================================================
*/
void Utils::getSafeSamplePoints(const Mat& img, const int& samplePointNum, const int& sampleSize, vector<Point>& pts) {
	
	// to draw samples from mask area
	if (img.channels() == 1) {
		Mat mask = img;
		vector<Point> ind;
		cv::findNonZero(img, ind);
		int nonZeros = static_cast<int>(ind.size());

		if (nonZeros > 0) {
			std::random_device random_device;
			std::mt19937 engine{ random_device() };
			std::uniform_int_distribution<int> dist(0, nonZeros - 1);

			int count = 0; // to record how many right sample points are found
			int iter = 0; // to record how many random points are tried out

			int N = nonZeros;
			if (nonZeros > samplePointNum) { N = samplePointNum; }

			std::set<pair<int, int>> new_ind;

			while (count < N) {
				Point  p = ind[dist(engine)];
				//check if the sample corners are on the border
				int x_min = p.x - int(sampleSize / 2); // (x,y) -> (col,row)
				int x_max = p.x + int(sampleSize / 2);
				int y_min = p.y - int(sampleSize / 2);
				int y_max = p.y + int(sampleSize / 2);
				// get rid of the points on the borders
				if (x_max < mask.cols && y_max < mask.rows && y_min >= 0 && x_min >= 0) {
					// check if the patch cross the class boarder
					Rect cell = Rect(x_min, y_min, sampleSize, sampleSize);
					if (cv::mean(mask(cell))[0] == mask.at<unsigned char>(y_min,x_min)) {
						new_ind.insert(pair<int, int>(p.x, p.y));
						count = new_ind.size();
					}
				}
				iter = iter + 1;
				if (iter > nonZeros) { break; }
			}

			for (auto it = new_ind.begin(); it != new_ind.end(); ++it)
			{
				pts.push_back(Point(it->first, it->second));
			}
		}
	}
	else {
		cout <<"please provide single channel mask or labelmap" << endl;
	}
}

// convert img data to png file
Mat Utils::convertDataToPNG(const Mat& src) {
	Mat dst;
	dst = src.clone();

	if (dst.channels() == 2) {
		dst = polsar::logTransform(dst);
	}
	else if (dst.channels() == 3) {
		cvtColor(dst, dst, COLOR_BGR2GRAY);
	}
	cv::normalize(dst, dst, 0, 255, cv::NORM_MINMAX);
	dst.convertTo(dst, CV_8UC1);
	cv::equalizeHist(dst, dst);
	return dst;
}

/************************************************************
Dividing the data samples into training and test samples
Take some training samples for each class and same for
test samples
Date: 11.06.2020
Modified by: Jun 15.06.2020
*************************************************************/
void Utils::DivideTrainTestData(const vector<Mat> &data, const vector<unsigned char> & data_label, int percentOfTrain,
	vector<Mat> & train_img,  vector<unsigned char> &train_label, vector<Mat>& test_img, vector<unsigned char> & test_label) {
	
	std::map<unsigned char, int> numPerClass;
	for (auto c : data_label) { numPerClass[c]++; }
	std::map<unsigned char, int> count;

	/*The idea is to get a balanced division between all the classes.
	5 classes with equal number of points. Also, the first 1/5th region is
	reserved for testing data set and from remaining area training samples are taken*/
	/*for each class*/
	for (int i = 0; i < data.size();i++ ) {
		unsigned char c = data_label[i];
		Mat img = data[i];
		++count[c];
		if (count[c] < numPerClass[c] * percentOfTrain / 100) {
			train_img.push_back(img);
			train_label.push_back(c);
		}
		else {
			test_img.push_back(img);
			test_label.push_back(c);
		}
	}
}

// shuffle the data
void Utils::shuffleDataSet(vector<Mat>& data, vector<unsigned char>& data_label) {
	int size = data.size();
	std::random_device random_device;
	std::mt19937 engine{ random_device() };
	std::uniform_int_distribution<int> rnd(0, size - 1);
	for (int i = 0; i < size; i++) {
		Mat temp = data[i];
		signed char temp_c = data_label[i];
		int swap = rnd(engine);
		if (swap == i) { continue; }
		else {
			data[i] = data[swap];
			data[swap] = temp;
			data_label[i] = data_label[swap];
			data_label[swap] = temp_c;
		}
	}
}

double Utils::calculatePredictionAccuracy(const vector<unsigned char>& classResult, const vector<unsigned char>& testLabels)
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

Mat Utils::getConfusionMatrix(const map<unsigned char, string>& className, vector<unsigned char>& classResult, vector<unsigned char>& testLabels) {
	map<pair<unsigned char, signed char>, int> testCount;

	for (int i = 0; i < testLabels.size(); i++) {
		for (int j = 0; j < classResult.size(); j++) {
			pair temp = make_pair(testLabels[i], classResult[j]);
			testCount[temp]++;
		}
	}

	int numOfClass = className.size();
	vector<unsigned char> classList(numOfClass);
	for (auto it = className.begin(); it != className.end(); it++) {
		classList.push_back(it->first);
	}

	Mat count = Mat(className.size(), className.size(), CV_8UC1);
	for (int i = 0; i < numOfClass; i++) {
		for (int j = 0; j < numOfClass; j++) {
			pair temp = make_pair(classList[i], classList[j]);
			count.at<unsigned char>(i, j) = testCount[temp];
		}
	}
	return count;
}

void Utils::deleteDataFromHDF(const String& filename, const String& parent_name, const vector<String>& dataset_name, int filterSize, int patchSize) {
	string parent = parent_name;

	if (filterSize != 0) { parent = parent + "_filterSize_" + to_string(filterSize); }
	for (int i = 0; i < dataset_name.size(); i++) {
		if (patchSize != 0) {
			deleteDataFromHDF(filename, parent, dataset_name[i] + "_patchSize_" + to_string(patchSize));
		}
		else {
			deleteDataFromHDF(filename, parent, dataset_name[i]);
		}
	}
}


void Utils::deleteDataFromHDF(const String& filename, const String& parent_name, const String& dataset_name) {
	Ptr<hdf::HDF5> h5io = hdf::open(filename);
	string datasetName = parent_name + dataset_name;

	if (!h5io->hlexists(parent_name)) {
		cout << parent_name << " is not existed." << endl;
	}else {
		if (!h5io->hlexists(datasetName)) {
			cout << datasetName << " is not existed." << endl;
		}else {
	        int result = h5io->dsdelete(datasetName);
			if (!result) {
				cout << "delete dataset " << datasetName << " success." << endl;
			}
			else {
				cout << "Failed to delete " << datasetName << endl;
			}
		}
	}
}



void Utils::writeDataToHDF(const String& filename, const String& parent_name, const vector<String>& dataset_name, const vector<Mat>& data, int filterSize , int patchSize) {
	string parent = parent_name;
	if (filterSize != 0) { parent = parent + "_filterSize_" + to_string(filterSize); }

	if (data.size() == dataset_name.size()) {
		for (int i = 0; i < data.size(); i++) {
			if(patchSize !=0){
			writeDataToHDF(filename, parent, dataset_name[i]+ "_patchSize_"+ to_string(patchSize), data[i]);
			}
			else {
				writeDataToHDF(filename, parent, dataset_name[i] , data[i]);
			}
		}
	}
	else {
		cout << "the size of dataset_name doesn't match that of data" << endl;
	}
}

void Utils::writeDataToHDF(const String& filename, const String& parent_name, const String& dataset_name, const Mat& src) {
	if(!src.empty()){
		Mat data = src.clone();
		if (data.channels() > 1) {
			for (size_t i = 0; i < data.total() * data.channels(); i++)
				((int*)data.data)[i] = (int)i;
		}

		Ptr<hdf::HDF5> h5io = hdf::open(filename);

		// first we need to create the parent group
		if (!h5io->hlexists(parent_name)) h5io->grcreate(parent_name);

		// create the dataset if it not exists
		string datasetName = parent_name + dataset_name;
		if (!h5io->hlexists(datasetName)) {
			h5io->dscreate(data.rows, data.cols, data.type(), datasetName);
			h5io->dswrite(data, datasetName);

			// check if the data are correctly write to hdf file
			Mat expected = Mat(Size(data.size()), data.type());
			h5io->dsread(expected, datasetName);
			double diff = norm(data - expected);
			CV_Assert(abs(diff) < 1e-10);

			if (h5io->hlexists(datasetName))
			{
				//cout << "write " << datasetName << " to " << filename << " success." << endl;
			}
			else {
				cout << "Failed to write " << datasetName << " to " << filename << endl;
			}
		}
		else {
			cout << datasetName << " is already existed." << endl;
		}
		h5io->close();
	}
}

void Utils::readDataFromHDF(const String& filename, const String& parent_name, const vector<String>& dataset_name, vector<Mat>& data, int filterSize, int patchSize) {
	string parent = parent_name;
	if (!data.empty()) { data.clear(); }

	if (filterSize != 0) { parent = parent + "_filterSize_" + to_string(filterSize); }
		for (int i = 0; i < dataset_name.size(); i++) {
			Mat temp;
			if (patchSize != 0) {
				readDataFromHDF(filename, parent, dataset_name[i] + "_patchSize_" + to_string(patchSize), temp);
			}
			else {
				readDataFromHDF(filename, parent, dataset_name[i], temp);
			}
			if (!temp.empty()) { data.push_back(temp); }
		}
}

void Utils::readDataFromHDF(const String& filename, const String& parent_name, const String& dataset_name, Mat& data) {

	Ptr<hdf::HDF5> h5io = hdf::open(filename);

	string datasetName = parent_name + dataset_name;

	if (!h5io->hlexists(parent_name)) {
		//cout << parent_name << " is not existed" << endl;
		data = Mat();
	}
	else if (!h5io->hlexists(datasetName) ) { 
		//cout << datasetName << " is not existed" << endl;  
		data = Mat(); 
	} else {
		vector<int> data_size = h5io->dsgetsize(datasetName);

		data = Mat(data_size[0],data_size[1],h5io->dsgettype(datasetName));

	    h5io->dsread(data, datasetName);
		//cout << "get " <<  datasetName  << " success" << endl;
	}

	h5io->close();
}


void Utils::writeAttrToHDF(const String& filename,const String& attribute_name,const int &attribute_value) {
	Ptr<hdf::HDF5> h5io = hdf::open(filename);
	if (!h5io->atexists(attribute_name)) {
		h5io->atwrite(attribute_value, attribute_name);
	}
	else {
		cout << " already existed" << endl;
	}
	h5io->close();
}

void Utils::writeAttrToHDF(const String& filename, const String& attribute_name, const string &attribute_value) {
	Ptr<hdf::HDF5> h5io = hdf::open(filename);
	if (!h5io->atexists(attribute_name)) {
		h5io->atwrite(attribute_value, attribute_name);
	}
	else {
		cout << " already existed" << endl;
	}
	h5io->close();
}

void Utils::readAttrFromHDF(const String& filename, const String& attribute_name,  string &attribute_value) {
	Ptr<hdf::HDF5> h5io = hdf::open(filename);
	if (!h5io->atexists(attribute_name)) {
		cout << attribute_name<<" is not existed" << endl;
	}
	else {
		h5io->atread(&attribute_value, attribute_name);
	}
	h5io->close();
}

void Utils::readAttrFromHDF(const String& filename, const String& attribute_name, int& attribute_value) {
	Ptr<hdf::HDF5> h5io = hdf::open(filename);
	if (!h5io->atexists(attribute_name)) {
		cout << attribute_name << " is not existed" << endl;
	}
	else {
		h5io->atread(&attribute_value, attribute_name);
	}
	h5io->close();
}

Mat Utils::readTiff(string filepath) {
	const char* file = filepath.c_str();
	GeoTiff* tiff = new GeoTiff(file);
	cout << "this tiff file has: cols " << tiff->NCOLS << " , rows " << tiff->NROWS << " , channels " << tiff->NLEVELS << endl;
	Mat data = tiff->GetMat().clone();
	delete tiff;
	return data;
}

bool Utils::checkExistInHDF(const String& filename, const String& parent_name, const string& dataset_name) {
	Ptr<hdf::HDF5> h5io = hdf::open(filename);
	bool flag = true;
	
	if (!h5io->hlexists(parent_name)) {
		flag = false;
		cout << parent_name << " is not existed" << endl;
	}else if (!h5io->hlexists(parent_name + dataset_name)) {
		flag = false;
		cout << parent_name + dataset_name << " is not existed" << endl;
	}
	h5io->close();
	return flag;
}


bool Utils::checkExistInHDF(const String& filename, const String& parent_name, const vector<string>& dataset_name, int filterSize, int patchSize ) {
	bool flag = true;
	string parent = parent_name;
	vector<string> dataset = dataset_name;
	if (filterSize != 0) {
		parent = parent_name + "_filterSize_" + to_string(filterSize);
	}

	for (auto& n : dataset) {
		if(patchSize !=0){
		   n =  n + "_patchSize_" + to_string(patchSize);
		}
		bool temp = checkExistInHDF(filename, parent, n);
		flag = flag && temp;
	}
	return	flag;
}


bool Utils::insertDataToHDF(const String& filename, const String& parent_name, const String& dataset_name, const Mat& data) {
	bool flag = true;
	if(!data.empty()){
		Ptr<hdf::HDF5> h5io = hdf::open(filename);

		if (checkExistInHDF(filename, parent_name, dataset_name)) {
			string dataset = parent_name + dataset_name;
			vector<int> data_size = h5io->dsgetsize(dataset);
			// expand the dataset at row direction
			int offset[2] = { data_size[0],0 };

			if ((h5io->dsgettype(dataset) == data.type()) && (data_size[1] == data.cols)) {
				h5io->dsinsert(data, dataset, offset);

				//check if insert success
				//cout << endl;
				//cout << "insert " << data.rows << " rows to" << dataset << " success " << endl;
				//cout << dataset << " rows in total: " << data.rows + offset[0] << endl;
			}

			else {
				flag = false;
				cout << endl;
				cout << " the new data has different size and type with the existed data" << endl;
				cout << dataset << " insert failed" << endl;
			}
		}
		else {
			// first we need to create the parent group
			if (!h5io->hlexists(parent_name)) h5io->grcreate(parent_name);

			string dataset = parent_name + dataset_name;
			int chunks[2] = { 1, data.cols };
			// create Unlimited x data.cols, data.type() space, dataset can be expanded unlimted on the row direction
			h5io->dscreate(hdf::HDF5::H5_UNLIMITED, data.cols, data.type(), dataset, hdf::HDF5::H5_NONE, chunks);
			// the first time to write data, offset at row,col direction is 0
			int offset[2] = { 0, 0 };
			h5io->dsinsert(data, dataset, offset);
			cout << endl;
			//cout << "insert " << data.rows << " rows to" << dataset << " success " << endl;
			//cout << dataset << " rows in total: " << data.rows + offset[0] << endl;
		}
	}
	return flag;
}

bool Utils::insertDataToHDF(const String& filename, const String& parent_name, const vector<string>& dataset_name, const vector<Mat>& data, int filterSize, int patchSize) {
	bool flag = true;
	string parent = parent_name;
	if (filterSize != 0) { parent = parent + "_filterSize_" + to_string(filterSize); }

	if (data.size() == dataset_name.size()) {
		for (int i = 0; i < data.size(); i++) {
			bool temp;
			if (patchSize != 0) {
				temp =insertDataToHDF(filename, parent, dataset_name[i] + "_patchSize_" + to_string(patchSize), data[i]);
			}
			else {
				temp =insertDataToHDF(filename, parent, dataset_name[i], data[i]);
			}
			flag = flag && temp;
		}
	}
	else {
		flag = false;
		cout << "the size of dataset_name doesn't match that of data"<<endl;
	}
	return flag;
}