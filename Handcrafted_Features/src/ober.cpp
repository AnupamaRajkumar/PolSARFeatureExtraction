#ifdef VC
#include <filesystem>
#endif // VC

#ifdef GCC
#include <dirent.h>
#endif
#include "ober.hpp" 


using namespace std;
using namespace cv;
namespace fs = std::filesystem;
 

/*===================================================================
 * Function: caculFeatures
 *
 * Summary:
 *   shuffle the samples and split them into batches with porper class distribution
 *	 calulate features and save to hdf5 file
 *
 * Arguments:
 *   const std::string& hdf5_fileName
 *   std::string feature_name - choose from { "texture", "color", "ctelements","polstatistic","decomp", "mp"}
 *   int filterSize - apply refined Lee despeckling filter, choose from (0, 5, 7, 9, 11)
 *	 int patchSize
 *   unsigned char classlabel  -  choose which class to load, default 255 means to load all the classes
 *	 int numOfSamplePoint - the number of samples for one type of class, default 0 means to load all the possible sample points
 *	 int batchSize - default 5000
 * output:
 *   
=====================================================================
*/
void ober::caculFeatures(const std::string& hdf5_fileName, const std::string& feature_name, int filterSize, int patchSize, int batchSize,int numOfSamplePoint, unsigned char classlabel) {
	
	std::map<string, int> feature_type = { {"texture",1},{"color",2},{"ctelements",3},{"polstatistic",4},{"decomp",5},{"mp",6} };
	vector<string> dataset_name = { "/feature" ,"/groundtruth" };
	std::string parent = "/" + feature_name + "_filterSize_" + std::to_string(filterSize) + "_patchSize_" + std::to_string(patchSize);

	writeLabelMapToHDF(hdf5_fileName, this->LabelMap, this->classNames);

	LoadSamplePoints(patchSize, numOfSamplePoint, classlabel, 1);

	cout << "start to calculate " << feature_name << " with filterSize " << filterSize << " , patchSize " << patchSize << endl;

	//shuffle all the samplePoints based on its label
	//make sure each class has right portion in each batch
	std::vector<std::vector<int>> subInd;
	Utils::splitVec(this->sampleLabel, subInd, batchSize);

	for (auto i = 0; i < subInd.size();i++) {
		std::vector<int> ind = subInd[i];
		size_t N = ind.size();

		vector<Mat> feature;
		vector<Mat> pts;
		for (size_t j = 0; j < N; ++j) {
			Point p = this->samplePoints[ind[j]];
			int patchLabel = this->sampleLabel[ind[j]];

			cv::Mat temp = Mat(1, 3, CV_32SC1);
			temp.at<int>(0, 0) = patchLabel;
			temp.at<int>(0, 1) = p.y; //row
			temp.at<int>(0, 2) = p.x; //col
			pts.push_back(temp);

			Mat hh, vv, hv;
			getSample(p, patchSize, filterSize, hh, vv, hv);

			switch (feature_type[feature_name]) {
			case 1:
				feature.push_back(caculTexture(hh, vv, hv));
				break;
			case 2:
				feature.push_back(caculColor(hh, vv, hv));
				break;
			case 3:
				feature.push_back(caculCTelements(hh, vv, hv));
				break;
			case 4:
				feature.push_back(caculPolStatistic(hh, vv, hv));
				break;
			case 5:
				feature.push_back(caculDecomp(hh, vv, hv));
				break;
			case 6:
				feature.push_back(caculMP(hh, vv, hv));
				break;
			default:
				std::cout << "feature name not existed:" << feature_name << std::endl;
				exit(-1);
				break;
			}
		}
		cv::Mat temp_feature, temp_pts;
		cv::vconcat(feature, temp_feature);
		cv::vconcat(pts, temp_pts);

		hdf5::insertData(hdf5_fileName, parent, dataset_name[0], temp_feature);
		hdf5::insertData(hdf5_fileName, parent, dataset_name[1], temp_pts);
		feature.clear();
		pts.clear();
		std::cout << "calculate "<<feature_name<<" progress: " << float(i+1) / float(subInd.size()) * 100.0 << "% \n" << std::endl;
	}
}


/*===================================================================
 * Function: LoadSamplePoints
 *
 * Summary:
 *   load sample points
 *
 * Arguments:
 *   const int& patchSize - choose from { "/texture", "/color", "/ctElements","/polStatistic","/decomp", "/MP"}
 *	 const int& numOfSamplePoint - the number of samples for one type of class, 0 means load all the possible sample points
 *   const unsigned char& classlabel - choose which class to load, 255 means to load all the classes
 *	 int stride
 * output:
 *
=====================================================================
*/
void ober::LoadSamplePoints(const int& patchSize, const int& numOfSamplePoint, const unsigned char& classlabel, int stride) {

	if (!this->samplePoints.empty()) {
		this->samplePoints.clear();
	}

	if (!this->sampleLabel.empty()) {
		this->sampleLabel.clear();
	}

	cout << "start to generate sample points with patchSize " << patchSize << ", stride " << stride << "..." << endl;

	for (const auto& classname : this->classNames) {
		unsigned char label = unsigned char(0);
		string name = classname.second;
		vector<Point> pts;
		if (classlabel == unsigned char(255)) {
			label = classname.first;
		}
		else if( classlabel == classname.first){
			label = classlabel;
		}
		else {
			continue;
		}
		 
		Utils::getRandomSamplePoint(this->LabelMap, pts, label, patchSize, stride, numOfSamplePoint);
		cout << "Get " << pts.size() << " sample points for class " << name << endl;
		for (size_t i = 0; i < pts.size(); i++) {
			this->sampleLabel.push_back(label);
		}
		copy(pts.begin(), pts.end(), back_inserter(this->samplePoints));
		pts.clear();
	}
}

/*===================================================================
 * Function: getSample
 *
 * Summary:
 *   get sample mat centered at sample point
 *   apply refined Lee filter to the sample mat if filtersize is not 0
 *
 * Arguments:
 *  const Point& p - sample point
 *	int patchSize
 *	int filtersize - choose from {0, 5, 7, 9, 11}
 * output:
 * Mat& hh, Mat& vv, Mat& hv
=====================================================================
*/
void ober::getSample(const Point& p, int patchSize, int filtersize, Mat& hh, Mat& vv, Mat& hv) {
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
	if (x_max < data[0].cols && y_max < data[0].rows && y_min >= 0 && x_min >= 0) {
		if (this->data.size() == 3) {
			if (filtersize == 5 || filtersize == 7 || filtersize == 9 || filtersize == 11) {
				hh = this->data[0](roi).clone();
				vv = this->data[1](roi).clone();
				hv = this->data[2](roi).clone();

				RefinedLee* filter = new RefinedLee(filtersize, 1);
				filter->filterFullPol(hh, vv, hv);
				delete filter;
			}
			else {
				hh = data[0](roi);
				vv = data[1](roi);
				hv = data[2](roi);
			}
		}
		else if (data.size() == 2) {
			vv = this->data[0](roi);
			hv = this->data[1](roi);
		}
	}
	else {
		cout << "out of boundary, get sample at point (" << p.x << "," << p.y << "with patchSize " << patchSize << " failed " << endl;
		hh = Mat();
		vv = Mat();
		hv = Mat();
	}
}

// calculate texture features on each channel
Mat ober::caculTexture(const Mat& hh, const Mat& vv, const Mat& hv) {
	vector<Mat> temp(3);
	// intensity of HH channel
	temp[0] = polsar::logTransform(polsar::getComplexAmpl(hh));

	// intensity of VV channel
	temp[1] = polsar::logTransform(polsar::getComplexAmpl(vv));

	// intensity of HV channel
	temp[2] = polsar::logTransform(polsar::getComplexAmpl(hv));

	vector<Mat> output;
	for (const auto& t : temp) {
		Mat result;
		hconcat(cvFeatures::GetGLCM(t, 8, GrayLevel::GRAY_8, 32), cvFeatures::GetLBP(t, 1, 8, 32), result);
		output.push_back(result);
	}

	Mat result;
	vconcat(output, result);
	return result.reshape(1,1);
}

// calculate color features on Pauli Color Coding
Mat ober::caculColor(const Mat& hh, const Mat& vv, const Mat& hv) {
	Mat colorImg = polsar::GetPauliColorImg(hh, vv, hv);
	Mat result;
	cv::hconcat(cvFeatures::GetMPEG7CSD(colorImg, 32), cvFeatures::GetMPEG7DCD(colorImg, 1), result);
	return result;
}

// calculate morphological profile on each channel with diameter (1,3,5)
Mat ober::caculMP(const Mat& hh, const Mat& vv, const Mat& hv) {
	 Mat hh_log = polsar::logTransform(polsar::getComplexAmpl(hh));
	 Mat vv_log = polsar::logTransform(polsar::getComplexAmpl(vv));
	 Mat hv_log = polsar::logTransform(polsar::getComplexAmpl(hv));
	 Mat result;
	 result.push_back(cvFeatures::GetMP(hh_log, { 1,3,5 }));
	 result.push_back(cvFeatures::GetMP(vv_log, { 1,3,5 }));
	 result.push_back(cvFeatures::GetMP(hv_log, { 1,3,5 }));

	 return result.reshape(1,1);
}

// calculate target decomposition
Mat ober::caculDecomp(const Mat& hh, const Mat& vv, const Mat& hv) {
	Mat result;

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

	vector<Mat> decomposition;
	polsar::GetCloudePottierDecomp(coherency, decomposition); //5  
	polsar::GetFreemanDurdenDecomp(coherency, decomposition); //3  
	polsar::GetKrogagerDecomp(circ, decomposition); // 3  
	polsar::GetPauliDecomp(pauli, decomposition); // 3  
	vconcat(decomposition, result);
	return result.reshape(1, 1);
}


// get polsar features on elements of covariance matrix C and coherency matrix T
Mat ober::caculCTelements(const Mat& hh, const Mat& vv, const Mat& hv) {
	Mat result;
	vector<Mat> temp;
	polsar::GetCTelements(hh, vv, hv, temp);

	for (auto& d : temp) {
		result.push_back(d);
	}

	return result.reshape(1, 1);

}


// get polsar features on statistic of polsar parameters
Mat ober::caculPolStatistic(const Mat& hh, const Mat& vv, const Mat& hv) {

	vector<Mat> temp;
	// intensity of HH channel
	Mat hh_log = polsar::logTransform(polsar::getComplexAmpl(hh));
	
	// intensity of VV channel
	Mat vv_log = polsar::logTransform(polsar::getComplexAmpl(vv));

	// intensity of HV channel
	Mat hv_log = polsar::logTransform(polsar::getComplexAmpl(hv));

	// phase difference HH-VV
	Mat phaseDiff = polsar::getPhaseDiff(hh, vv);

	//statistic of Co-polarize ratio VV-HH
	Mat coPolarize = vv_log - hh_log;

	// Cross-polarized ratio HV-HH
	Mat crossPolarize = hv_log - hh_log;

	// polarized ratio HV-VV
	Mat otherPolarize = hv_log - vv_log;

	//Copolarization ratio
	Mat copolarizationRatio = polsar::getCoPolarizationRatio(hh, vv, 3);

	//deCopolarization ratio
	Mat deCopolarizationRatio = polsar::getDePolarizationRatio(hh, vv, hv, 3);

	// amplitude of HH-VV correlation
	Mat amplitudeCorrelation = polsar::logTransform(polsar::calcuCoherenceOfPol(hh, vv, 3)) - hh_log - vv_log;

	temp.push_back(hh_log);
	temp.push_back(vv_log);
	temp.push_back(hv_log);
	temp.push_back(phaseDiff);
	temp.push_back(coPolarize);
	temp.push_back(crossPolarize);
	temp.push_back(otherPolarize);
	temp.push_back(copolarizationRatio);
	temp.push_back(deCopolarizationRatio);
	temp.push_back(amplitudeCorrelation);

	vector<Mat> statistic;
	for (const auto& t : temp) {
		statistic.push_back(cvFeatures::GetStatistic(t));
	}

	Mat output;
	cv::hconcat(statistic, output);
	return output.reshape(1, 1);
}


void ober::writeLabelMapToHDF(const string& hdf5_fileName,  Mat& labelmap, std::map<unsigned char, std::string>& classnames) {

	string parent_name = "/masks";
	
	if(!hdf5::checkExist(hdf5_fileName,parent_name, "/labelMap")){
		// save labelMap to hdf5
		hdf5::writeData(hdf5_fileName, parent_name, "/labelMap", labelmap);
		cout << "write labelMap to hdf5 success " << endl;

		// save the class name to hdf5
		cv::Mat classlabels;
		for (auto& name : classnames) {
			classlabels.push_back(name.first);
			hdf5::writeAttr(hdf5_fileName,to_string(name.first), name.second);
		}
		hdf5::writeAttr(hdf5_fileName, "classlabels", classlabels.reshape(1,1));
	}

	// save the intensity of HH,VV,HV to hdf5
	if (!hdf5::checkExist(hdf5_fileName, parent_name, "/hh_intensity")) {
		cout << "write the intensity of HH, VV, HV to hdf5 success " << endl;
		Mat hh = polsar::logTransform(this->data[0]);
		Mat vv = polsar::logTransform(this->data[1]);
		Mat hv = polsar::logTransform(this->data[2]);
		hdf5::writeData(hdf5_fileName, parent_name, "/hh_intensity", hh);
		hdf5::writeData(hdf5_fileName, parent_name, "/vv_intensity", vv);
		hdf5::writeData(hdf5_fileName, parent_name, "/hv_intensity", hv);
	}
}



void ober::getSampleInfo(const string& hdf5_fileName,const Mat& pts, int patchSize) {
	cout << "it has " << pts.rows << " samples with patchSize " << patchSize << endl;
	map<int, int> count;
	for (int row = 0; row < pts.rows; row++) {
		int label = pts.at<int>(row, 0);
		count[label]++;
	}
	cout << "class name (unknown, 0) means this patch cross class boarder or unclassified" << endl;
	for (auto const& c : count)
	{
		int label = c.first;
		int sampleNum = c.second;
		string class_name;
		hdf5::readAttr(hdf5_fileName, "label_" + to_string(label), class_name);
		cout << class_name <<" : "<< to_string(label) <<" : number of samples: " << sampleNum << endl;
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

	//check if it is rat file
	size_t pos = 0;
	if (fname.find("rat", pos) == std::string::npos) {
		cout << " this is not rat file" << endl;
		exit(-1);
	}

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
	for (auto i = 0; i < dim; i++) {
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
		for (auto i = 0; i < dim - 1; i++) cout << imgSize[i] << " x ";
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
		img.convertTo(img, CV_32FC1);
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