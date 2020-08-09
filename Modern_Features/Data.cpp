#include "Data.h"

#include <iostream>
#include <fstream>
#ifdef VC
#include <filesystem>
#endif // VC

#ifdef GCC
#include <dirent.h>
#endif

#include <random>


using namespace std;
using namespace cv;
namespace fs = std::filesystem;


// the default constructor
Data::Data(void) {
	labelImages.reserve(NUMOFCLASSES);
	labelNames.reserve(NUMOFCLASSES);
	numOfPoints.reserve(NUMOFCLASSES);
}

// destructor
Data::~Data(void) {
	// nothing to do
}


void Data::getTileInfo(cv::Size size, unsigned border, unsigned &tile, vector<unsigned> &tileSize, vector<unsigned> &tileStart) {
	
	cout << "Size:" << size.width << "x" << size.height << endl;				//1390 x 6640; number of channels:3
	tileSize[0] = size.width;
	tileSize[1] = size.height;

	tileStart[0] = 0;
	tileStart[1] = 0;

}
/**************************************************************
Function to load Oberpfaffenhofen PolSAR data file (RAT format
***************************************************************/

void Data::loadPolSARData(std::vector<std::string> const& fname) {

	switch (fname.size()) {
	case 1: {        // one rat file with scattering vector or matrix
		loadRAT(fname[0], this->data);
		break;
	}
	case 2: {        // dual-pol, one file per channel
		vector<Mat> ch1, ch2;
		loadRAT(fname[0], ch1);
		loadRAT(fname[1], ch2);
		this->data.push_back(ch1[0]);
		this->data.push_back(ch2[0]);
		break;
	}
	case 3: {        // full-pol, averaged cross-pol, one file per channel
		vector<Mat> hh, vv, xx;
		loadRAT(fname[0], hh);
		loadRAT(fname[1], vv);
		loadRAT(fname[2], xx);
		this->data.push_back(hh[0]);
		this->data.push_back(vv[0]);
		this->data.push_back(xx[0]);
		break;
	}
	case 4: {        // full-pol, individual cross-pol, one file per channel
		vector<Mat> hh, vv, hv, vh;
		loadRAT(fname[0], hh);
		loadRAT(fname[1], vv);
		loadRAT(fname[2], hv);
		loadRAT(fname[2], vh);
		this->data.push_back(hh[0]);
		this->data.push_back(vv[0]);
		this->data.push_back(0.5*(hv[0] + vh[0]));
		break;
	}
	}
}
/**************************************************************
Function to load Oberpfaffenhofen PolSAR data file (RAT format)
***************************************************************/
Size Data::loadRAT2(string fname, vector<Mat>& data, bool metaOnly) {

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

	char *buf = new char(dsize);
	char *swap = new char(dsize);
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
					data[i*dim[0] + j] = Mat::zeros(tileSize[1], tileSize[0], CV_32FC1);
				else
					data[i*dim[0] + j] = Mat::zeros(tileSize[1], tileSize[0], CV_32FC2);
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
							data.at(j*dim[1] + i).at<float>(dim[3] - y - 1 - tileStart[1], x - tileStart[0]) = realVal;
						else
							data.at(j*dim[1] + i).at<Vec2f>(dim[3] - y - 1 - tileStart[1], x - tileStart[0]) = Vec2f(realVal, imagVal);
					}
				}
			}
		}
		break;
	}
	return Size(dim[ndim - 2], dim[ndim - 1]);
}

/**************************************************************
Function to load Oberpfaffenhofen PolSAR data file (RAT format)
***************************************************************/

Size Data::loadRAT(string fname, vector<Mat> &data, bool metaOnly) {

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
	char *buf = new char(dsize);
	char *swap = new char(dsize);
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
					data[i*imgSize[0] + j] = Mat::zeros(tileSize[1], tileSize[0], CV_32FC1);
				else
					data[i*imgSize[0] + j] = Mat::zeros(tileSize[1], tileSize[0], CV_32FC2);
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
							data.at(j*imgSize[1] + i).at<float>(imgSize[3] - y - 1 - tileStart[1], x - tileStart[0]) = realVal;
						else
							data.at(j*imgSize[1] + i).at<Vec2f>(imgSize[3] - y - 1 - tileStart[1], x - tileStart[0]) = Vec2f(realVal, imagVal);
					}
				}
			}
		}
		break;
	}
	return Size(imgSize[dim - 2], imgSize[dim - 1]);
}

/**************************************************************
Author: Anupama Rajkumar
Description: Function to load Oberpfaffenhofen PolSAR image file 
Input : image file name
***************************************************************/
Mat Data::loadImage(string fname) {
	Mat polSARData = imread(fname, 0);

	if (!polSARData.data) {
		cout << "ERROR: Cannot find the original image" << endl;
		cout << "Press enter to exit.." << endl;
		cin.get();
		exit(0);
	}

	return polSARData.clone();
}


/**************************************************************
Function to read labels from label directory
Author: Anupama Rajkumar
Description : This function reads all the label images from the 
label directory
Input 1 : Label files directory path
Input 2 : Names of the label files
Output	: Label images found in the label directory
***************************************************************/
#ifdef VC

void Data::ReadClassLabels(string labelPath, vector<string> &labelNames, vector<Mat> &labelImages){
fs::recursive_directory_iterator iter(labelPath);
fs::recursive_directory_iterator end;
	while (iter != end) {
		string tp = iter->path().string();
		size_t pos = 0;
		//get the filename from path without extension
		//string base_filename = tp.substr(tp.find_last_of("/\\") + 1);
		string base_filename = tp.substr(tp.find_last_of("/\\") + 1);
		size_t position = base_filename.find(".");
		string fileName = (string::npos == position) ? base_filename : base_filename.substr(0, position);
		labelNames.push_back(fileName);
		Mat img = imread(iter->path().string(), 1);
		
		if (!img.data) {
			cout << "ERROR: Cannot find labelled image" << endl;
			cout << "Press enter to exit..." << endl;
			cin.get();
			exit(0);
		}
		/*Ensure that the number of channels is always 1*/
		if (img.channels() > 1) {
			cvtColor(img, img, COLOR_BGR2GRAY);
		}
		img.convertTo(img, CV_32FC1);
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

void Data::ReadClassLabels(string labelPath, vector<string> &labelNames, vector<Mat> &labelImages) {
	struct dirent *entry;
	DIR *dir = opendir(labelPath.c_str());

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

/*************************************************************************
Function to load labels
Author: Anupama Rajkumar
Description : The function loads the labels and counts the number of
points for each label class
Input 1  : Label folder path
Output 1 : Label names
Output 2 : Label images
Output 3 : Number of non zero points for each label class found in the 
		   label directory
************************************************************************/
void Data::loadLabels(const string &folderPath, vector<string>& labelNames, vector<Mat>& labelImages,
					 vector<vector<Point2i>>& numOfPoints) {

	this->ReadClassLabels(folderPath, labelNames, labelImages);
	/*Loading label points*/
	for (int cnt = 0; cnt < labelImages.size(); cnt++) {
		vector<Point2i> classPoints;
		classPoints.reserve(labelImages[cnt].rows*labelImages[cnt].cols);
		for (int row = 0; row < labelImages[cnt].rows; row++) {
			for (int col = 0; col < labelImages[cnt].cols; col++) {
				if (labelImages[cnt].at<float>(row, col) > 0.0f) {
					Point2i newPoint(row, col);
					classPoints.push_back(newPoint);
				}
			}
		}
		numOfPoints.push_back(classPoints);
	}
	//print the names of label files
	for (int cnt = 0; cnt < labelImages.size(); cnt++)
	{
		cout << labelNames[cnt] << " " << labelImages[cnt].cols << " x " << labelImages[cnt].rows << " \t " << numOfPoints[cnt].size()<< endl;
	}
}

/***************************************************************************
Function to load PolSAR data
Author: Anupama Rajkumar
Description : This function loads the PolSAR file from the respective directory
Input 1  : PolSAR data folder path
****************************************************************************/

#ifdef VC

void Data::loadData(string folderPath) {
	vector<string> fnames;
	fnames.reserve(5);

	fs::recursive_directory_iterator iter(folderPath);
	fs::recursive_directory_iterator end;
	while (iter != end) {
		string p = iter->path().string();
		fnames.push_back(iter->path().string());
		error_code ec;
		iter.increment(ec);
		if (ec) {
			std::cerr << "Error while accessing:" << iter->path().string() << "::" << ec.message() << "\n";
		}
	}
	this->loadPolSARData(fnames);
}

#endif	//VC

