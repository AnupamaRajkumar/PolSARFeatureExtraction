#include "ober.hpp" 
#include "cvFeatures.hpp"
#include "sarFeatures.hpp"
#include <opencv2/opencv.hpp>

#ifdef VC
#include <filesystem>
#endif // VC

#ifdef GCC
#include <dirent.h>
#endif
#include "cv_hdf5.hpp"
#include <complex>
#include <string>
#include <iostream>
#include <fstream>
#include <random>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;
 
// write calculated features to hdf5 ( sample points, labels, features)
void ober::saveFeaturesToHDF(const String& hdf5_fileName, const String& parent_name, const vector<String>& dataset_name, vector<Mat>& features, vector<unsigned char>& featureLabels, int filterSize, int patchSize){
	Mat pts = Mat(this->samplePoints.size(), 3, CV_32SC1);
	for (size_t i = 0; i < this->samplePoints.size(); i++) {
		pts.at<int>(i, 0) = (int)(featureLabels[i]);
		pts.at<int>(i, 1) = this->samplePoints[i].x; //row
		pts.at<int>(i, 2) = this->samplePoints[i].y; //col
	}

	Mat result;
	for (auto& f : features) {
		Mat temp = f.clone();
		result.push_back(temp.reshape(1,1));
	}
	  Utils::insertDataToHDF(hdf5_fileName, parent_name, dataset_name, { result,pts }, filterSize, patchSize);
	/* if(Utils::checkExistInHDF(hdf5_fileName,parent_name,dataset_name, filterSize, patchSize)){
	  features.clear();
	  featureLabels.clear();
	 }*/
}

// get features from hdf5
void ober::getFeaturesFromHDF(const String& hdf5_fileName, const String& parent_name, const vector<String>& dataset_name, vector<Mat>& features, vector<unsigned char>& featureLabels, int filterSize, int patchSize) {
	vector<Mat> data;
	if (Utils::checkExistInHDF(hdf5_fileName, parent_name, dataset_name, filterSize, patchSize)) {
		Utils::readDataFromHDF(hdf5_fileName, parent_name, dataset_name, data, filterSize, patchSize);
		Mat result = data[0];
		Mat pts = data[1]; //labelPoint
		for (int i = 0; i < result.rows; i++) {
			features.push_back(result.row(i));
			featureLabels.push_back((unsigned char)(pts.at<int>(i, 0)));
		}
	}
}


void ober::LoadSamplePoints(const int &patchSize, const int & numOfSamplePoint) {

	this->sampleSize = patchSize;

	getSamplePoints(this->hdf5_file, this->samplePoints,this->samplePointClassLabel,patchSize, numOfSamplePoint);
}

// load customized sample points
void ober::LoadSamplePoints(const vector<Point>& points, const vector<unsigned char>& pointLabels, const int& patchSize) {
	this->samplePoints = points;
	this->samplePointClassLabel = pointLabels;
	this->sampleSize = patchSize;
	map<unsigned char, int> count;
	for (auto& l : pointLabels) {
		count[l] ++;
	}
	for (auto& c : count) {
		cout << "Get " << c.second<< " sample points for class " << classNames[c.first] << endl;
	}
}


// get patches of 3 channel (HH+VV,HV,HH-VV) intensity(dB)
void ober::GetPauliColorPatches(vector<Mat>& patches, vector<unsigned char>& classValue) {
	
	if (!samplePoints.empty()) {
			for (const auto& p : samplePoints) {
				Mat hh, vv, hv;
				getSample(p, this->sampleSize, this->filterSize, hh, vv, hv);

				patches.push_back(polsar::GetPauliColorImg(hh, vv, hv));
		}
			classValue = samplePointClassLabel;
	}
}


// get patches of 3 channel (HH,VV,HV) intensity(dB)
void ober::GetPatches(vector<Mat>& patches, vector<unsigned char>& classValue) {
		 
		if(!samplePoints.empty()){
				for (const auto& p : samplePoints) {
					Mat hh, vv, hv;
					getSample(p, this->sampleSize, this->filterSize, hh, vv, hv);
					
					patches.push_back(polsar::GetFalseColorImg(hh, vv, hv, Mat(),false));
				}
				classValue = samplePointClassLabel;
		}
}


// get texture features(LBP and GLCM) on HH,VV,VH 
void ober::GetTextureFeature(vector<Mat>& features, vector<unsigned char>& classValue) {

		std::map< unsigned char, int> count;
		cout << "start to draw texture features ... " <<endl;

		for (int i = 0; i < samplePoints.size();i++) {
			Point p = samplePoints.at(i);

			//cout << classNames[samplePointClassLabel.at(i)] << " :draw texture feature at Point (" << p.x << ", " << p.y << ")" << endl;

			Mat hh, vv, hv;
			getSample(p, this->sampleSize, this->filterSize, hh, vv, hv);

			vector<Mat> temp;
			// intensity of HH channel
			  hh = polsar::logTransform(polsar::getComplexAmpl(hh));
			// intensity of VV channel
			  vv = polsar::logTransform(polsar::getComplexAmpl(vv));
			// intensity of HV channel
			  hv = polsar::logTransform(polsar::getComplexAmpl(hv));

			temp.push_back(hh);
			temp.push_back(vv);
			temp.push_back(hv);

			vector<Mat> output;
			for (const auto& t : temp) {
				Mat result;
				hconcat(cvFeatures::GetGLCM(t, 8, GrayLevel::GRAY_8, 32), cvFeatures::GetLBP(t, 1, 8, 32), result);
				output.push_back(result);
			}

			Mat result;
			vconcat(output, result);
			features.push_back(result);
			classValue.push_back(samplePointClassLabel.at(i));

			count[samplePointClassLabel.at(i)]++;
		}

		/*for (auto it = count.begin(); it != count.end(); ++it)
		{
			std::cout << "get "<< it->second  <<" texture features for class " <<  classNames[it->first] << std::endl;
		}*/
}



// get color features(MPEG-7 DCD,CSD) on Pauli Color image, default feature mat size 1*44
void ober::GetColorFeature(vector<Mat>& features, vector<unsigned char>& classValue) {
	std::map< unsigned char, int> count;
	cout << "start to draw color features ... " << endl;

	for (int i = 0; i < samplePoints.size(); i++) {
		Point p = samplePoints.at(i);

		//cout << classNames[samplePointClassLabel.at(i)] << " :draw color feature at Point (" << p.y << ", " << p.x << ")" << endl;
		
		Mat hh, vv, hv;
		getSample(p, this->sampleSize, this->filterSize, hh, vv, hv);

		Mat colorImg = polsar::GetPauliColorImg(hh, vv, hv);

		Mat result;
		cv::hconcat(cvFeatures::GetMPEG7CSD(colorImg, 32), cvFeatures::GetMPEG7DCD(colorImg, 3), result);
		features.push_back(result);
		classValue.push_back(samplePointClassLabel.at(i));

		count[samplePointClassLabel.at(i)]++;
	}

	/*for (auto it = count.begin(); it != count.end(); ++it)
	{
		std::cout << "get " << it->second << " color features for class " << classNames[it->first] << std::endl;
	}*/
}


// get MP features on HH,VV,VH, default feature mat size (sampleSize*9,sampleSize)
void ober::GetMPFeature(vector<Mat>& features, vector<unsigned char>& classValue) {
	std::map< unsigned char, int> count;
	cout << "start to draw MP features ... " << endl;

	for (int i = 0; i < samplePoints.size(); i++) {
		Point p = samplePoints.at(i);

		//cout << classNames[samplePointClassLabel.at(i)] << " :draw MP feature at Point (" << p.y << ", " << p.x << ")" << endl;

		Mat hh, vv, hv;
		getSample(p, this->sampleSize, this->filterSize, hh, vv, hv);


		vector<Mat> temp(3);
		// intensity of HH channel
		temp[0] = polsar::logTransform(polsar::getComplexAmpl(hh));
		// intensity of VV channel
		temp[1] = polsar::logTransform(polsar::getComplexAmpl(vv));
		// intensity of HV channel
		temp[2]= polsar::logTransform(polsar::getComplexAmpl(hv));

		Mat result;
		for (const auto& t : temp) {
			result.push_back(cvFeatures::GetMP(t, { 1,3,5 }));
		}
		features.push_back(result);
		classValue.push_back(samplePointClassLabel.at(i));

		count[samplePointClassLabel.at(i)]++;
	}

	/*for (auto it = count.begin(); it != count.end(); ++it)
	{
		std::cout << "get " << it->second << " mp features for class " << classNames[it->first] << std::endl;
	}*/
}


// get polsar features on target decompostion 
void ober::GetDecompFeatures(vector<Mat>& features, vector<unsigned char>& classValue) {
	std::map< unsigned char, int> count;
	cout << "start to draw target decomposition features... " << endl;

	for (int i = 0; i < samplePoints.size(); i++) {
		Point p = samplePoints.at(i);

		//cout << classNames[samplePointClassLabel.at(i)] << " :draw target decomposition feature at Point (" << p.y << ", " << p.x << ")" << endl;

		Mat hh, vv, hv;
		getSample(p, this->sampleSize, this->filterSize, hh, vv, hv);

		//caculate the decomposition of sample patch
		Mat result;
		vector<Mat> decomposition;
		getTargetDecomposition(hh, vv, hv, decomposition);
		vconcat(decomposition, result);

		features.push_back(result);
		classValue.push_back(samplePointClassLabel.at(i));

		count[samplePointClassLabel.at(i)]++;
	}

	/*for (auto it = count.begin(); it != count.end(); ++it)
	{
		std::cout << "get " << it->second << " target decomposition features for class " << classNames[it->first] << std::endl;
	}*/
}

// get polsar features on elements of covariance matrix C and coherency matrix T
void ober::GetCTFeatures(vector<Mat>& features, vector<unsigned char>& classValue) {
	std::map< unsigned char, int> count;
	cout << "start to draw matrix C and T elements ... " << endl;

	for (int i = 0; i < samplePoints.size(); i++) {
		Point p = samplePoints.at(i);

		//cout << classNames[samplePointClassLabel.at(i)] << " :draw matrix C and T elements at Point (" << p.y << ", " << p.x << ")" << endl;

		Mat hh, vv, hv;
		getSample(p, this->sampleSize, this->filterSize, hh, vv, hv);

		//caculate the decomposition at sample point
		Mat result;
		vector<Mat> decomposition;
		polsar::GetCTelements(hh, vv, hv, decomposition);

		for (auto& d : decomposition) {
			result.push_back(d);
		}

		features.push_back(result);
		classValue.push_back(samplePointClassLabel.at(i));

		count[samplePointClassLabel.at(i)]++;
	}

	/*for (auto it = count.begin(); it != count.end(); ++it)
	{
		std::cout << "get " << it->second << " CT elements features for class " << classNames[it->first] << std::endl;
	}*/
}

// get polsar features on statistic of polsar parameters
void ober::GetPolsarStatistic(vector<Mat>& features, vector<unsigned char>& classValue) {
	std::map< unsigned char, int> count;
	cout << "start to draw statistic features of polsar parameters ... " << endl;

	for (int i = 0; i < samplePoints.size(); i++) {
		Point p = samplePoints.at(i);

		//cout << classNames[samplePointClassLabel.at(i)] << " :draw statistic polarimetric feature at Point (" << p.y << ", " << p.x << ")" << endl;

		Mat hh, vv, hv;
		getSample(p, this->sampleSize,this->filterSize, hh, vv, hv);

		Mat result;
		vector<Mat> statistic;
		polsar::GetFullPolStat(hh, vv, hv, statistic);
		cv::hconcat(statistic, result);

		 
		features.push_back(result);
		classValue.push_back(samplePointClassLabel.at(i));

		count[samplePointClassLabel.at(i)]++;
	}

	/*for (auto it = count.begin(); it != count.end(); ++it)
	{
		std::cout << "get " << it->second << " statistic polarimetric features for class " << classNames[it->first] << std::endl;
	}*/
}




// get target decomposition features
// vector<mat>& result - vector length: , mat size: (hh.size())
void ober::getTargetDecomposition(const Mat& hh, const Mat& vv, const Mat &hv, vector<Mat>& result) {
	vector<Mat> pauli;
	vector<Mat> circ;
	vector<Mat> lexi;
	polsar::getPauliBasis(hh, vv, hv, pauli);
	polsar::getCircBasis(hh, vv, hv, circ);
	polsar::getLexiBasis(hh, vv, hv, lexi);
	vector<Mat> covariance;
	vector<Mat> coherency;
	polsar::GetCoherencyT(pauli, coherency);
	polsar::GetCovarianceC(lexi, covariance);

	//polsar::GetCloudePottierDecomp(coherency, result); //3  
	polsar::GetFreemanDurdenDecomp(coherency, result); //3  
	//polsar::GetKrogagerDecomp(circ, result); // 3  
	//polsar::GetPauliDecomp(pauli, result); // 3  
	//polsar::GetHuynenDecomp(covariance, result); // 9  
	//polsar::GetYamaguchi4Decomp(coherency, covariance, result); //4  
}

//// apply despeckling filter, choose from ( 5, 7, 9, 11)
//void ober::applyRefinedLeeFilter(int filterSize) {
//	//check if filtered data is already existed
//
//	if (filterSize == 5 || filterSize == 7 || filterSize == 9 || filterSize == 11) {
//		RefinedLee* filter = new RefinedLee(filterSize, 1);
//
//		if (data.size() == 3) {
//			cout << "start to apply refined lee filter with filterSize " << filterSize << " ..." << endl;
//			filter->filterFullPol(data[0], data[1], data[2]);
//		}
//		else {
//			cout << "now only support full pol data" << endl;
//		}
//
//		delete filter;
//
//	}
//}


// get data at sample point
void ober::getSample(const Point& p,int patchSize, int filtersize,Mat& hh, Mat& vv, Mat& hv) {
	int size = patchSize;
	int start_x = int(p.x) - patchSize / 2;
	int start_y = int(p.y) - patchSize / 2;
	Rect roi = Rect(start_x, start_y, size, size);
	
	//boundary check
	//check if the sample corners are on the border
	int x_min = p.x - int(patchSize / 2); // (x,y) -> (col,row)
	int x_max = p.x + int(patchSize / 2);
	int y_min = p.y - int(patchSize / 2);
	int y_max = p.y + int(patchSize / 2);
	if (x_max < data[0].cols && y_max < data[0].rows && y_min >= 0 && x_min >= 0){
		if (this->data.size() == 3) {
			hh = this->data[0](roi).clone();
			vv = this->data[1](roi).clone();
			hv = this->data[2](roi).clone();

			if (filtersize == 5 || filtersize == 7 || filtersize == 9 || filtersize == 11) {
				RefinedLee* filter = new RefinedLee(filtersize, 1);
				filter->filterFullPol(hh, vv, hv);
				delete filter;
			}
		}
		else if(data.size() ==2) {
			vv = this->data[0](roi);
			hv = this->data[1](roi);
		}
	}
	else {
		cout << "out of boundary, get sample at point (" << p.x << "," << p.y << "with patchSize "<< patchSize <<" failed " << endl;
		hh = Mat();
		vv = Mat();
		hv = Mat();
	}
}

void ober::writeLabelMapToHDF(const string& hdf5_fileName) {
	string parent_name = "/masks";
	Mat labelMap = Utils::generateLabelMap(this->masks);

	Utils::writeDataToHDF(hdf5_fileName, parent_name, "/labelMap", labelMap);

	Utils::writeAttrToHDF(hdf5_fileName, "label_" + to_string(0), "unknown");

	for (auto & name: this->classNames) {
		Utils::writeAttrToHDF(hdf5_fileName, "label_" + to_string(name.first), name.second);
	}
}


void ober::generateSamplePoints(const string& hdf5_fileName, int patchSize) {
	
	int stride = 1;
	string parent_name = "/samplePoints";
	string data_name = "/patchSize_" + to_string(patchSize);
	bool flag = Utils::checkExistInHDF(hdf5_fileName, parent_name, data_name);

	Mat samplePoints;
	if (!flag) {

		cout << "\n" << endl;
		cout << "start to generate sample points with patchSize " << patchSize << ", stride " << stride << "..." << endl;

		Mat labelMap;
		Utils::readDataFromHDF(hdf5_file, "/masks", "/labelMap", labelMap);
		if (labelMap.empty()) {
			writeLabelMapToHDF(hdf5_fileName);
			Utils::readDataFromHDF(hdf5_fileName, "/masks", "/labelMap", labelMap);
		}
		
			for (int row = 0; row < labelMap.rows - patchSize; row += stride) {
				for (int col = 0; col < labelMap.cols - patchSize; col += stride) {
					Rect cell = Rect(col, row, patchSize, patchSize);

					Mat temp;
					int halfsize = patchSize / 2;

					int label = 0;
					// if this patch cross differenct class boarder, set the label to 0 (unclassified)
					int row_max = row + patchSize - 1;
					int col_max = col + patchSize - 1;
					// check if the cornor of this patch stay at same class area
					if (labelMap.at<unsigned char>(row, col_max) == labelMap.at<unsigned char>(row, col) &&
						labelMap.at<unsigned char>(row_max, col) == labelMap.at<unsigned char>(row, col) &&
						labelMap.at<unsigned char>(row_max, col_max) == labelMap.at<unsigned char>(row, col))
					{
						label = int(labelMap.at<unsigned char>(row, col));
					}
					//record the central points of each patch
					temp.push_back(label);
					temp.push_back(row + patchSize / 2);
					temp.push_back(col + patchSize / 2);

					samplePoints.push_back(temp.reshape(1, 1));
				}
			}
			Utils::writeDataToHDF(hdf5_fileName, parent_name, data_name, samplePoints);

			cout << "get " << samplePoints.rows << " samples with patch size " << patchSize << endl;
			cout << "class_name: unknown, label: 0, indicates the sample crosses class boarder or unlabelled" << endl;
			getSamplePointInfo( hdf5_fileName,samplePoints, patchSize);
	}
	else {
		cout << "" << endl;
		cout << parent_name + data_name << " is already in " << hdf5_fileName << endl;
		Utils::readDataFromHDF(hdf5_fileName, parent_name, data_name, samplePoints);
		getSamplePointInfo( hdf5_fileName,samplePoints, patchSize);
	}
}

void ober::getSamplePointInfo(const string& hdf5_fileName,const Mat& pts, int patchSize) {
	cout << "it has " << pts.rows << " sample points with patchSize " << patchSize << endl;
	map<int, int> count;
	for (int row = 0; row < pts.rows; row++) {
		int label = pts.at<int>(row, 0);
		count[label]++;
	}
	for (auto const& c : count)
	{
		int label = c.first;
		int sampleNum = c.second;
		string class_name;
		Utils::readAttrFromHDF(hdf5_fileName, "label_" + to_string(label), class_name);
		cout << "class_name: " << class_name << ", label: " << label << " number of samples: " << sampleNum << endl;
	}

}

// get random sample points 
void ober::getSamplePoints(const string& hdf5_fileName, vector<Point> & samplePoints, vector<unsigned char> &sampleLabels, int patchSize, int numOfSamplePoints) {
	cout << endl;
	
	bool flag = Utils::checkExistInHDF(hdf5_fileName, "/samplePoints", "/patchSize_" + to_string(patchSize));
	if (!flag) {
		generateSamplePoints(hdf5_fileName,patchSize);
	}
	Mat pts;
	Utils::readDataFromHDF(hdf5_fileName, "/samplePoints", "/patchSize_" + to_string(patchSize), pts);

		// get the num of points for each class
		map<int, vector<int>> count;
		for (int row = 0; row < pts.rows; row++) {
			int label = pts.at<int>(row, 0);
			if (label != 0) { count[label].push_back(row); }
		}
		// for each class
		// loop through each label class
		cout << "" << endl;

		for (auto const& c : count)
		{
			int label = c.first;
			vector<int> rows = c.second;
			string class_name;
			Utils::readAttrFromHDF(hdf5_fileName, "label_" + to_string(label), class_name);

			//cout << "class_name: " << class_name << ", label: " << label << " number of samples: " << c.second.size() << endl;
			size_t num = 0;
			std::random_device random_device;
			std::mt19937 engine{ random_device() };
			std::uniform_int_distribution<int> pt(0, rows.size() - 1);
			while (num < numOfSamplePoints) {
				int i = rows[pt(engine)];
				Point p;
				p.x = pts.at<int>(i, 2);
				p.y = pts.at<int>(i, 1);
				samplePoints.push_back(p);
				sampleLabels.push_back((unsigned char)(pts.at<int>(i, 0)));
				++num;
			}
			cout << "draw " << num << " sample points for class " << class_name << endl;
		}
	
}

#ifdef VC
void ober::loadData(string RATfolderPath) {
	vector<string> fnames;
	fnames.reserve(5);

	fs::recursive_directory_iterator iter(RATfolderPath);
	fs::recursive_directory_iterator end;
	while (iter != end) {
		string tmp = iter->path().string();


		fnames.push_back(tmp);
		error_code ec;
		iter.increment(ec);
		if (ec) {
			std::cerr << "Error while accessing:" << iter->path().string() << "::" << ec.message() << "\n";
		}
	}

	switch (fnames.size()) {
	case 1: {        // one rat file with scattering vector or matrix
		loadRAT(fnames[0], this->data);
		break;
	}
	case 2: {        // dual-pol, one file per channel
		vector<Mat> ch1, ch2;
		loadRAT(fnames[0], ch1);
		loadRAT(fnames[1], ch2);
		this->data.push_back(ch1[0]);
		this->data.push_back(ch2[0]);
		break;
	}
	case 3: {        // full-pol, averaged cross-pol, one file per channel
		vector<Mat> hh, vv, xx;
		loadRAT(fnames[0], hh);
		loadRAT(fnames[1], vv);
		loadRAT(fnames[2], xx);
		this->data.push_back(hh[0]);
		this->data.push_back(vv[0]);
		this->data.push_back(xx[0]);
		break;
	}
	case 4: {        // full-pol, individual cross-pol, one file per channel
		vector<Mat> hh, vv, hv, vh;
		loadRAT(fnames[0], hh);
		loadRAT(fnames[1], vv);
		loadRAT(fnames[2], hv);
		loadRAT(fnames[2], vh);
		this->data.push_back(hh[0]);
		this->data.push_back(vv[0]);
		this->data.push_back(0.5 * (hv[0] + vh[0]));
		break;
	}
	}
}


#endif


/**************************************************************
Function to load Oberpfaffenhofen PolSAR data file (RAT format)
***************************************************************/
Size ober::loadRAT2(string fname, vector<Mat>& data, bool metaOnly) {

	bool verbose = false;

	// header info
	int magiclong;
	float version;
	int ndim;
	int nchannel;
	int dim[8];
	int var;
	int sub[2];
	int type;
	int reserved[9];

	// open file
	fstream file(fname.c_str(), ios::in | ios::binary);
	if (!file)
		cerr << "ERROR: Cannot open file: " << fname << endl;

	// read header
	file.read((char*)(&magiclong), sizeof(magiclong));
	file.read((char*)(&version), sizeof(version));
	file.read((char*)(&ndim), sizeof(ndim));
	file.read((char*)(&nchannel), sizeof(nchannel));
	file.read((char*)(&dim), sizeof(dim));
	file.read((char*)(&var), sizeof(var));
	file.read((char*)(&sub), sizeof(sub));
	file.read((char*)(&type), sizeof(type));
	file.read((char*)(&reserved), sizeof(reserved));

	if (verbose) {
		cout << "Number of image dimensions:\t" << ndim << endl;
		cout << "Image dimensions:\t";
		for (int i = 0; i < ndim - 1; i++)
			cout << dim[i] << " x ";
		cout << dim[ndim - 1] << endl;
		cout << "Data type:\t" << var << endl;
		cout << "Type:\t" << type << endl;
	}

	if (metaOnly) {
		file.close();
		return Size(dim[ndim - 2], dim[ndim - 1]);
	}

	vector<unsigned> tileSize(2);
	unsigned tile;
	vector<unsigned> tileStart(2);
	this->getTileInfo(Size(dim[ndim - 2], dim[ndim - 1]), this->border, tile, tileSize, tileStart);

	if (verbose) {
		cout << "Tile:\t\t" << tile << endl;
		cout << "Tile size (cols x rows):\t" << tileSize[0] << "x" << tileSize[1] << endl;
		cout << "Tile start (col x row):\t" << tileStart[0] << "x" << tileStart[1] << endl;
	}

	file.seekg(0);
	file.seekg(1000);
	int nChannels = 0, dsize = 0;
	switch (var) {
	case 1:
		nChannels = 1;
		dsize = 1;
		break;
	case 2:
		nChannels = 1;
		dsize = 4;
		break;
	case 3:
		nChannels = 1;
		dsize = 4;
		break;
	case 4:
		nChannels = 1;
		dsize = 4;
		break;
	case 5:
		nChannels = 1;
		dsize = 8;
		break;
	case 12:
		nChannels = 1;
		dsize = 4;
		break;
	case 13:
		nChannels = 1;
		dsize = 4;
		break;
	case 14:
		nChannels = 1;
		dsize = 8;
		break;
	case 15:
		nChannels = 1;
		dsize = 8;
		break;
	case 6:
		nChannels = 2;
		dsize = 4;
		break;
	case 9:
		nChannels = 2;
		dsize = 8;
		break;
	default: cerr << "ERROR: arraytyp not recognized (wrong format?)" << endl;
	}

	char* buf = new char(dsize);
	char* swap = new char(dsize);
	int i, j, x, y;
	Mat img, real, imag;
	switch (ndim) {
	case 2:
		data.resize(1);
		if (nChannels == 1)
			data[0] = Mat::zeros(tileSize[1], tileSize[0], CV_32FC1);
		else
			data[0] = Mat::zeros(tileSize[1], tileSize[0], CV_32FC2);
		for (y = 0; y < dim[1]; y++) {
			for (x = 0; x < dim[0]; x++) {
				double realVal, imagVal;
				file.read((char*)(&buf), dsize);
				for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
				switch (var) {
				case 1:
					dsize = 1;
					realVal = *((char*)buf);
					break;	// byte
				case 2:
					dsize = 4;
					realVal = *((int*)buf);
					break;	// int
				case 3:
					dsize = 4;
					realVal = *((long*)buf);
					break;	// long
				case 4:
					dsize = 4;
					realVal = *((float*)buf);
					break;	// float
				case 5:
					dsize = 8;
					realVal = *((double*)buf);
					break;	// double
				case 6:
					dsize = 4;					// complex
					realVal = *((float*)buf);
					file.read((char*)(&buf), dsize);
					for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
					imagVal = *((float*)buf);
					break;
				case 9:
					dsize = 8;					// dcomplex
					realVal = *((double*)buf);
					file.read((char*)(&buf), dsize);
					for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
					imagVal = *((double*)buf);
					break;
				case 12:
					dsize = 4;
					realVal = *((unsigned int*)buf);
					break;	// uint
				case 13:
					dsize = 4;
					realVal = *((unsigned long*)buf);
					break;	// ulong
				case 14:
					dsize = 4;
					realVal = *((double*)buf);
					break;	// l64
				case 15:
					dsize = 4;
					realVal = *((double*)buf);
					break;	// ul64
				}
				if ((dim[1] - y - 1 < tileStart[1]) || (dim[1] - y - 1 >= tileStart[1] + tileSize[1])) continue;
				if ((x < tileStart[0]) || (x >= tileStart[0] + tileSize[0])) continue;
				if (nChannels != 2)
					data[0].at<float>(dim[1] - y - 1 - tileStart[1], x - tileStart[0]) = realVal;
				else
					data[0].at<Vec2f>(dim[1] - y - 1 - tileStart[1], x - tileStart[0]) = Vec2f(realVal, imagVal);
			}
		}
		break;
	case 3:
		data.resize(dim[0]);
		for (i = 0; i < dim[0]; i++) {
			if (nChannels == 1)
				data[i] = Mat::zeros(tileSize[1], tileSize[0], CV_32FC1);
			else
				data[i] = Mat::zeros(tileSize[1], tileSize[0], CV_32FC2);
		}
		for (y = 0; y < dim[2]; y++) {
			for (x = 0; x < dim[1]; x++) {
				for (i = 0; i < dim[0]; i++) {
					double realVal, imagVal;
					file.read((char*)(&buf), dsize);
					for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
					switch (var) {
					case 1:
						dsize = 1;
						realVal = *((char*)buf);
						break;	// byte
					case 2:
						dsize = 4;
						realVal = *((int*)buf);
						break;	// int
					case 3:
						dsize = 4;
						realVal = *((long*)buf);
						break;	// long
					case 4:
						dsize = 4;
						realVal = *((float*)buf);
						break;	// float
					case 5:
						dsize = 8;
						realVal = *((double*)buf);
						break;	// double
					case 6:
						dsize = 4;					// complex
						realVal = *((float*)buf);
						file.read((char*)(&buf), dsize);
						for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
						imagVal = *((float*)buf);
						break;
					case 9: dsize = 8;					// dcomplex
						realVal = *((double*)buf);
						file.read((char*)(&buf), dsize);
						for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
						imagVal = *((double*)buf);
						break;
					case 12:
						dsize = 4;
						realVal = *((unsigned int*)buf);
						break;	// uint
					case 13:
						dsize = 4;
						realVal = *((unsigned long*)buf);
						break;	// ulong
					case 14:
						dsize = 4;
						realVal = *((double*)buf);
						break;	// l64
					case 15:
						dsize = 4;
						realVal = *((double*)buf);
						break;	// ul64
					}
					if ((dim[2] - y - 1 < tileStart[1]) || (dim[2] - y - 1 >= tileStart[1] + tileSize[1])) continue;
					if ((x < tileStart[0]) || (x >= tileStart[0] + tileSize[0])) continue;
					if (nChannels != 2)
						data.at(i).at<float>(dim[2] - y - 1 - tileStart[1], x - tileStart[0]) = realVal;
					else
						data.at(i).at<Vec2f>(dim[2] - y - 1 - tileStart[1], x - tileStart[0]) = Vec2f(realVal, imagVal);
				}
			}
		}
		break;
	case 4:
		data.resize(dim[0] * dim[1]);
		for (i = 0; i < dim[0]; i++) {
			for (j = 0; j < dim[1]; j++) {
				if (nChannels == 1)
					data[i * dim[0] + j] = Mat::zeros(tileSize[1], tileSize[0], CV_32FC1);
				else
					data[i * dim[0] + j] = Mat::zeros(tileSize[1], tileSize[0], CV_32FC2);
			}
		}
		for (y = 0; y < dim[3]; y++) {
			for (x = 0; x < dim[2]; x++) {
				for (j = 0; j < dim[0]; j++) {
					for (i = 0; i < dim[1]; i++) {
						double realVal, imagVal;
						file.read((char*)(&buf), dsize);
						for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
						switch (var) {
						case 1:
							dsize = 1;
							realVal = *((char*)buf);
							break;	// byte
						case 2:
							dsize = 4;
							realVal = *((int*)buf);
							break;	// int
						case 3:
							dsize = 4;
							realVal = *((long*)buf);
							break;	// long
						case 4:
							dsize = 4;
							realVal = *((float*)buf);
							break;	// float
						case 5:
							dsize = 8;
							realVal = *((double*)buf);
							break;	// double
						case 6: dsize = 4;					// complex
							realVal = *((float*)buf);
							file.read((char*)(&buf), dsize);
							for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
							imagVal = *((float*)buf);
							break;
						case 9: dsize = 8;					// dcomplex
							realVal = *((double*)buf);
							file.read((char*)(&buf), dsize);
							for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
							imagVal = *((double*)buf);
							break;
						case 12:
							dsize = 4;
							realVal = *((unsigned int*)buf);
							break;	// uint
						case 13:
							dsize = 4;
							realVal = *((unsigned long*)buf);
							break;	// ulong
						case 14:
							dsize = 4;
							realVal = *((double*)buf);
							break;	// l64
						case 15:
							dsize = 4;
							realVal = *((double*)buf);
							break;	// ul64
						}
						if ((dim[3] - y - 1 < tileStart[1]) || (dim[3] - y - 1 >= tileStart[1] + tileSize[1])) continue;
						if ((x < tileStart[0]) || (x >= tileStart[0] + tileSize[0])) continue;
						if (nChannels != 2)
							data.at(j * dim[1] + i).at<float>(dim[3] - y - 1 - tileStart[1], x - tileStart[0]) = realVal;
						else
							data.at(j * dim[1] + i).at<Vec2f>(dim[3] - y - 1 - tileStart[1], x - tileStart[0]) = Vec2f(realVal, imagVal);
					}
				}
			}
		}
		break;
	}
	return Size(dim[ndim - 2], dim[ndim - 1]);
}

/**************************************************************
Function to load Oberpfaffenhofen PolSAR data file (RAT format
***************************************************************/

Size ober::loadRAT(string fname, vector<Mat>& data, bool metaOnly) {

	bool verbose = true;

	// header info
	unsigned int dim;
	vector<unsigned int> imgSize;
	unsigned int var;
	unsigned int type;
	unsigned int dummy;
	char info[80];

	// open file
	fstream file(fname.c_str(), ios::in | ios::binary);
	if (!file) {
		cout << "ERROR: Cannot open file: " << fname << endl;
		exit(-1);
	}

	// read header
	file.read((char*)(&dim), sizeof(dim));
	dim = (dim >> 24) | ((dim << 8) & 0x00FF0000) | ((dim >> 8) & 0x0000FF00) | (dim << 24);

	if (dim > 1000) {
		return loadRAT2(fname, data, metaOnly);
	}

	imgSize.resize(dim);
	for (int i = 0; i < dim; i++) {
		file.read((char*)(&imgSize[i]), sizeof(imgSize[i]));
		imgSize[i] = (imgSize[i] >> 24) | ((imgSize[i] << 8) & 0x00FF0000) | ((imgSize[i] >> 8) & 0x0000FF00) | (imgSize[i] << 24);
	}
	file.read((char*)(&var), sizeof(var));
	var = (var >> 24) | ((var << 8) & 0x00FF0000) | ((var >> 8) & 0x0000FF00) | (var << 24);
	file.read((char*)(&type), sizeof(type));
	type = (type >> 24) | ((type << 8) & 0x00FF0000) | ((type >> 8) & 0x0000FF00) | (type << 24);
	file.read((char*)(&dummy), sizeof(dummy));
	file.read((char*)(&dummy), sizeof(dummy));
	file.read((char*)(&dummy), sizeof(dummy));
	file.read((char*)(&dummy), sizeof(dummy));
	file.read(info, sizeof(info));

	if (verbose) {
		cout << "Number of image dimensions:\t" << dim << endl;
		cout << "Image dimensions:\t";
		for (int i = 0; i < dim - 1; i++) cout << imgSize[i] << " x ";
		cout << imgSize[dim - 1] << endl;
		cout << "Data type:\t" << var << endl;
		cout << "Type:\t" << type << endl;
		cout << "Info:\t" << info << endl;
	}

	if (metaOnly) {
		file.close();
		return Size(imgSize[dim - 2], imgSize[dim - 1]);
	}

	vector<unsigned> tileSize(2);
	unsigned tile;
	vector<unsigned> tileStart(2);
	this->getTileInfo(Size(imgSize[dim - 2], imgSize[dim - 1]), this->border, tile, tileSize, tileStart);

	if (verbose) {
		cout << "Tile:\t\t" << tile << endl;
		cout << "Tile size (cols x rows):\t" << tileSize[0] << "x" << tileSize[1] << endl;
		cout << "Tile start (col x row):\t" << tileStart[0] << "x" << tileStart[1] << endl;
	}
	int nChannels = 0, dsize = 0;
	switch (var) {
	case 1:
		nChannels = 1;
		dsize = 1;
		break;
	case 2:
		nChannels = 1;
		dsize = 4;
		break;
	case 3:
		nChannels = 1;
		dsize = 4;
		break;
	case 4:
		nChannels = 1;
		dsize = 4;
		break;
	case 5:
		nChannels = 1;
		dsize = 8;
		break;
	case 12:
		nChannels = 1;
		dsize = 4;
		break;
	case 13:
		nChannels = 1;
		dsize = 4;
		break;
	case 14:
		nChannels = 1;
		dsize = 8;
		break;
	case 15:
		nChannels = 1;
		dsize = 8;
		break;
	case 6:										//comes here - Oberpfaffenhofen
		nChannels = 2;
		dsize = 4;
		break;
	case 9:
		nChannels = 2;
		dsize = 8;
		break;
	default: cerr << "ERROR: arraytyp not recognized (wrong format?)" << endl;
		exit(-1);
	}
	char* buf = new char(dsize);
	char* swap = new char(dsize);
	int i, j, x, y;
	Mat img;
	switch (dim) {
	case 2:         // scalar SAR image (e.g. only magnitude)
		data.resize(1);
		if (nChannels == 1)
			data[0] = Mat::zeros(tileSize[1], tileSize[0], CV_32FC1);
		else
			data[0] = Mat::zeros(tileSize[1], tileSize[0], CV_32FC2);
		for (y = 0; y < imgSize[1]; y++) {
			for (x = 0; x < imgSize[0]; x++) {
				//file.read((char*)(&buf), dsize);
				file.read(buf, dsize);
				double realVal, imagVal;
				// swap number
				for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
				switch (var) {
				case 1:
					realVal = *((char*)swap);
					break;	// byte
				case 2:
					realVal = *((int*)swap);
					break;	// int
				case 3:
					realVal = *((long*)swap);
					break;	// long
				case 4:
					realVal = *((float*)swap);
					break;	// float
				case 5:
					realVal = *((double*)swap);
					break;	// double
				case 6:
					realVal = *((float*)swap);
					//file.read((char*)(&buf), dsize);
					file.read(buf, dsize);
					for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
					imagVal = *((float*)swap);
					break;
				case 9:
					realVal = *((double*)swap);
					file.read((char*)(&buf), dsize);
					for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
					imagVal = *((double*)swap);
					break;
				case 12:
					realVal = *((unsigned int*)swap);
					break;	// uint
				case 13:
					realVal = *((unsigned long*)swap);
					break;	// ulong
				case 14:
					realVal = *((double*)swap);
					break;	// l64
				case 15:
					realVal = *((double*)swap);
					break;	// ul64
				}
				if ((imgSize[1] - y - 1 < tileStart[1]) || (imgSize[1] - y - 1 >= tileStart[1] + tileSize[1])) continue;
				if ((x < tileStart[0]) || (x >= tileStart[0] + tileSize[0])) continue;
				if (nChannels != 2)
					data[0].at<float>(imgSize[1] - y - 1 - tileStart[1], x - tileStart[0]) = realVal;
				else
					data[0].at<Vec2f>(imgSize[1] - y - 1 - tileStart[1], x - tileStart[0]) = Vec2f(realVal, imagVal);
			}
		}
		break;
	case 3:         // 3D SAR image (e.g. scattering vector)				//comes here - oberpfaffenhofen
		data.resize(imgSize[0]);
		for (i = 0; i < imgSize[0]; i++) {
			if (nChannels == 1)
				data[i] = Mat::zeros(tileSize[1], tileSize[0], CV_32FC1);
			else
				data[i] = Mat::zeros(tileSize[1], tileSize[0], CV_32FC2);
		}
		for (y = 0; y < imgSize[2]; y++)
		{
			for (x = 0; x < imgSize[1]; x++)
			{
				for (i = 0; i < imgSize[0]; i++)
				{
					//file.read((char*)(&buf), dsize);
					file.read(buf, dsize);
					double realVal, imagVal;
					for (int d = 0; d < dsize; d++)
						swap[d] = buf[dsize - d - 1];
					switch (var) {
					case 1:
						realVal = *((char*)swap);
						break;	// byte
					case 2:
						realVal = *((int*)swap);
						break;	// int
					case 3:
						realVal = *((long*)swap);
						break;	// long
					case 4:
						realVal = *((float*)swap);
						break;	// float
					case 5:
						realVal = *((double*)swap);
						break;	// double
					case 6:
						realVal = *((float*)swap);			// complex
						//file.read((char*)(&buf), dsize);
						file.read(buf, dsize);							//comes here..oberpffafenhofen
						for (int d = 0; d < dsize; d++)
							swap[d] = buf[dsize - d - 1];
						imagVal = *((float*)swap);
						break;
					case 9:
						realVal = *((double*)swap);					// dcomplex
						file.read((char*)(&buf), dsize);
						for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
						imagVal = *((double*)swap);
						break;
					case 12:
						realVal = *((unsigned int*)swap);
						break;	// uint
					case 13:
						realVal = *((unsigned long*)swap);
						break;	// ulong
					case 14:
						realVal = *((double*)swap);
						break;	// l64
					case 15:
						realVal = *((double*)swap);
						break;	// ul64
					}
					if ((imgSize[2] - y - 1 < tileStart[1]) || (imgSize[2] - y - 1 >= tileStart[1] + tileSize[1])) continue;
					if ((x < tileStart[0]) || (x >= tileStart[0] + tileSize[0])) continue;
					if (nChannels != 2)
						data.at(i).at<float>(imgSize[2] - y - 1 - tileStart[1], x - tileStart[0]) = realVal;
					else
						data.at(i).at<Vec2f>(imgSize[2] - y - 1 - tileStart[1], x - tileStart[0]) = Vec2f(realVal, imagVal);
				}
			}
		}
		break;
	case 4:             // 4D SAR image (e.g. scattering matrix)
		data.resize(imgSize[0] * imgSize[1]);
		for (i = 0; i < imgSize[0]; i++) {
			for (j = 0; j < imgSize[1]; j++) {
				if (nChannels == 1)
					data[i * imgSize[0] + j] = Mat::zeros(tileSize[1], tileSize[0], CV_32FC1);
				else
					data[i * imgSize[0] + j] = Mat::zeros(tileSize[1], tileSize[0], CV_32FC2);
			}
		}
		for (y = 0; y < imgSize[3]; y++) {
			for (x = 0; x < imgSize[2]; x++) {
				for (j = 0; j < imgSize[0]; j++) {
					for (i = 0; i < imgSize[1]; i++) {
						file.read((char*)(&buf), dsize);
						double realVal, imagVal;
						for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
						switch (var) {
						case 1:
							dsize = 1; realVal = *((char*)swap);
							break;	// byte
						case 2:
							dsize = 4; realVal = *((int*)swap);
							break;	// int
						case 3:
							dsize = 4; realVal = *((long*)swap);
							break;	// long
						case 4:
							dsize = 4; realVal = *((float*)swap);
							break;	// float
						case 5:
							dsize = 8; realVal = *((double*)swap);
							break;	// double
						case 6:
							dsize = 4;					// complex
							realVal = *((float*)swap);
							file.read((char*)(&buf), dsize);
							for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
							imagVal = *((float*)swap);
							break;
						case 9:
							dsize = 8;					// dcomplex
							realVal = *((double*)swap);
							file.read((char*)(&buf), dsize);
							for (int d = 0; d < dsize; d++) swap[d] = buf[dsize - d - 1];
							imagVal = *((double*)swap);
							break;
						case 12:
							dsize = 4; realVal = *((unsigned int*)swap);
							break;	// uint
						case 13:
							dsize = 4; realVal = *((unsigned long*)swap);
							break;	// ulong
						case 14:
							dsize = 4; realVal = *((double*)swap);
							break;	// l64
						case 15:
							dsize = 4; realVal = *((double*)swap);
							break;	// ul64
						}
						if ((imgSize[3] - y - 1 < tileStart[1]) || (imgSize[3] - y - 1 >= tileStart[1] + tileSize[1])) continue;
						if ((x < tileStart[0]) || (x >= tileStart[0] + tileSize[0])) continue;
						if (nChannels != 2)
							data.at(j * imgSize[1] + i).at<float>(imgSize[3] - y - 1 - tileStart[1], x - tileStart[0]) = realVal;
						else
							data.at(j * imgSize[1] + i).at<Vec2f>(imgSize[3] - y - 1 - tileStart[1], x - tileStart[0]) = Vec2f(realVal, imagVal);
					}
				}
			}
		}
		break;
	}
	return Size(imgSize[dim - 2], imgSize[dim - 1]);
}

/**************************************************************
Function to read labels from label directory
Author: Anupama Rajkumar
Date: 23.05.2020
***************************************************************/
#ifdef VC

void ober::ReadClassLabels(string labelPath, vector<string>& labelNames, vector<Mat>& labelImages) {
	fs::recursive_directory_iterator iter(labelPath);
	fs::recursive_directory_iterator end;
	while (iter != end) {
		string tp = iter->path().string();
		size_t pos = 0;
		//get the filename from path without extension
		string base_filename = tp.substr(tp.find_last_of("/\\") + 1);
		size_t position = base_filename.find(".");
		string fileName = (string::npos == position) ? base_filename : base_filename.substr(0, position);
		labelNames.push_back(fileName);

		Mat img = imread(tp);

		if (!img.data) {
			cout << "ERROR: Cannot find labelled image" << endl;
			cout << "Press enter to exit..." << endl;
			cin.get();
			exit(0);
		}
		// mask file should be 1 channel
		if (img.channels() > 1) {
			cvtColor(img, img, COLOR_BGR2GRAY);
		}
		else
		{
			img.convertTo(img, CV_8UC1);
		}
		labelImages.push_back(img);

		error_code ec;
		iter.increment(ec);
		if (ec) {
			std::cerr << "Error while accessing:" << iter->path().string() << "::" << ec.message() << "\n";
		}
	}
}
#endif // VC

#ifdef GCC

void ober::ReadClassLabels(string labelPath, vector<string>& labelNames, vector<Mat>& labelImages) {
	struct dirent* entry;
	DIR* dir = opendir(labelPath.c_str());

	if (dir == NULL)
		return;

	std::size_t current = 0;
	int i = 0;
	while ((entry = readdir(dir)) != NULL)
	{
		if (strlen(entry->d_name) < 10)
			continue; // Ignore current folder (.) and parent folder (..)

		labelNames.push_back(entry->d_name);
		string filename = labelPath + labelNames[i];
		Mat_<float> img = imread(filename, IMREAD_GRAYSCALE);
		if (!img.data)
		{
			cout << "ERROR: file " << filename << " not found" << endl;
			cout << "Press enter to exit" << endl;
			cin.get();
			exit(-3);
		}
		// convert to floating point precision
		img.convertTo(img, CV_64FC1);
		labelImages.push_back(img);
		cout << "Loaded image " << labelNames[i] << endl;
		i++;
	}
	closedir(dir);
}

#endif //GCC

void ober::getTileInfo(cv::Size size, unsigned border, unsigned& tile, vector<unsigned>& tileSize, vector<unsigned>& tileStart) {

	cout << "Size:" << size.width << "x" << size.height << endl;				//1390 x 6640; number of channels:3
	tileSize[0] = size.width;
	tileSize[1] = size.height;

	tileStart[0] = 0;
	tileStart[1] = 0;

}