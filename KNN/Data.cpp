#include "Data.h"

#include <iostream>
#include <fstream>
#include <filesystem>
#include <random>


using namespace std;
using namespace cv;
namespace fs = std::filesystem;


// the default constructor
Data::Data(void) {

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
Function to load Oberpfaffenhofen PolSAR data file (RAT format
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
Function to load Oberpfaffenhofen PolSAR image file 
Author: Anupama Rajkumar
Date: 23.05.2020
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
	//cout << polSARData.channels();
	//show the image
	//imshow("Image", polSARData);
}


/**************************************************************
Function to read labels from label directory
Author: Anupama Rajkumar
Date: 23.05.2020
***************************************************************/
void ReadClassLabels(string labelPath, vector<string> &labelNames, vector<Mat> &labelImages){
fs::recursive_directory_iterator iter(labelPath);
fs::recursive_directory_iterator end;
	while (iter != end) {
		labelNames.push_back(iter->path().string());

		Mat img = imread(iter->path().string());
		if (!img.data) {
			cout << "ERROR: Cannot find labelled image" << endl;
			cout << "Press enter to exit..." << endl;
			cin.get();
			exit(0);
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

/**************************************************************
Function to load labels
Author: Anupama Rajkumar
Date: 23.05.2020
***************************************************************/
void Data::loadLabels(const string &folderPath, vector<Mat>& labelImages, vector<string>& labelNames, vector<vector<Point2i>>& numOfPoints) {

	ReadClassLabels(folderPath, labelNames, labelImages);
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

/**************************************************************
Function to load PolSAR data
Author: Anupama Rajkumar
Date: 23.05.2020
***************************************************************/

void Data::loadData(string folderPath) {
	vector<string> fnames;
	fnames.reserve(5);

	fs::recursive_directory_iterator iter(folderPath);
	fs::recursive_directory_iterator end;
	while (iter != end) {
		fnames.push_back(iter->path().string());
		error_code ec;
		iter.increment(ec);
		if (ec) {
			std::cerr << "Error while accessing:" << iter->path().string() << "::" << ec.message() << "\n";
		}
	}
	this->loadPolSARData(fnames);
}


/***********************************************************************
Extracting patches from labels dataset
Author : Anupama Rajkumar
Date : 27.05.2020
*************************************************************************/

void Data::ExtractLabelPatches(vector<vector<Point2i>> numOfPoints, int numOfSamples, int sizeOfPatch, vector<Mat> labelImages, vector<vector<Point2i>>& samplePoints) {
	for (int cnt = 0; cnt < labelImages.size(); cnt++) {
		if (numOfPoints[cnt].size() < numOfSamples) {
			cout << "Class " << cnt + 1 << "does not have enough points to be drawn" << endl;
			vector<Point2i> emptyPoint;
			samplePoints.push_back(emptyPoint);
		}
		cout << "Drawing " << numOfSamples << "from " << numOfPoints[cnt].size() << "points from the class" << endl;
		vector<Point2i> classSamples = DrawSamples(numOfPoints[cnt], numOfSamples, sizeOfPatch, labelImages[cnt]);
		samplePoints.push_back(classSamples);
	}
}

/***********************************************************************
Draw random samples from labels
Author : Anupama Rajkumar
Date : 27.05.2020
*************************************************************************/

vector<Point2i> Data::DrawSamples(vector<Point2i> numOfPoints, int numOfSamples, int sizeOfPatch, Mat labelImages) {
	vector<Point2i> samples;
	samples.reserve(numOfSamples);
	int samplesDrawn = 0;

	//random samples generator
	std::random_device rd;										   // obtain a random number from hardware
	std::mt19937 eng(rd());										   // seed the generator
	std::uniform_int_distribution<> distr(0, numOfPoints.size()); // define the range

	while (samplesDrawn < numOfSamples)
	{
		int i = distr(eng);
		cout << "i" << i << endl;
		Point2i newSample(numOfPoints[i].x, numOfPoints[i].y);

		// Make sure that the samples point is not on a border.
		// This is needed in order to extract the patches.
		if (newSample.x < sizeOfPatch || newSample.y < sizeOfPatch ||
			newSample.x > labelImages.rows - sizeOfPatch || newSample.y > labelImages.cols - sizeOfPatch)
			continue;

		samples.push_back(newSample);
		samplesDrawn += 1;
	}
	return samples;
}

/***********************************************************************
Extract image patches around the label sample points
Author : Anupama Rajkumar
Date : 27.05.2020
*************************************************************************/
void Data::ExtractImagePatches(int numOfSamples, int sizeOfPatch, Mat RGBImg, vector<vector<Point2i>> samplePoints, vector<vector<Mat>>& patches) {

	for (int cnt = 0; cnt < NUMOFCLASSES; cnt++) {
		cout << "Getting patches for " << cnt + 1 << "label class" << endl;
		vector<Mat> imgPatches;
		imgPatches.reserve(numOfSamples);
		imgPatches = GetPatches(RGBImg, samplePoints[cnt], sizeOfPatch);
		patches.push_back(imgPatches);
	}
}

/***********************************************************************
Extract image patches around the label sample points
Author : Anupama Rajkumar
Date : 27.05.2020
*************************************************************************/
vector<Mat> Data::GetPatches(Mat origImg, vector<Point2i> samplePoints, int sizeOfPatch)
{
	vector<Mat> patches;
	patches.reserve(samplePoints.size());
	for (int cnt = 0; cnt < samplePoints.size(); cnt++)
	{
		Mat_<Vec3f> newPatch(sizeOfPatch, sizeOfPatch);
		int pStart_r = samplePoints[cnt].x - sizeOfPatch / 2;
		int pStart_c = samplePoints[cnt].y - sizeOfPatch / 2;

		int pEnd_r = samplePoints[cnt].x + sizeOfPatch / 2;
		int pEnd_c = samplePoints[cnt].y + sizeOfPatch / 2;

		if (pStart_r < 0 || pStart_c < 0 || pEnd_r > origImg.rows || pEnd_c > origImg.cols)
		{
			cout << "Patch lies outside the image boundary" << endl;
			continue;
		}
		//cout << "Extracting patches of size " << samplePoints.size() << endl;
		//form patches
		int r, row;
		int c, col;
		row = 0;
		col = 0;
		for (r = pStart_r; r < pEnd_r; r++)
		{
			col = 0;
			for (c = pStart_c; c < pEnd_c; c++)
			{
				newPatch.at<Vec3f>(row, col) = origImg.at<Vec3f>(r, c);
				col++;
			}
			row++;
		}
		patches.push_back(newPatch);
	}
	return patches;
}

/***********************************************************************
Extract image points extraction
Author : Anupama Rajkumar
Date : 28.05.2020
*************************************************************************/
void Data::ExtractImagePoints(int numOfSamples, Mat& RGBImg, vector<Point2i>& samples) {

	//vector<Point2i> samples;
	//samples.reserve(numOfSamples);
	int samplesDrawn = 0;

	//random samples generator
	std::random_device rd;										   // obtain a random number from hardware
	std::mt19937 eng(rd());										   // seed the generator
	std::uniform_int_distribution<> distrX(0, RGBImg.rows);		   // define the range
	std::uniform_int_distribution<> distrY(0, RGBImg.cols);

	while (samplesDrawn < numOfSamples)
	{
		int x = distrX(eng);
		int y = distrY(eng);
		Point2i newSample(x, y);
		samples.push_back(newSample);
		samplesDrawn += 1;
	}
}


