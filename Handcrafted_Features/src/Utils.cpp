#include "Utils.h"
#include <filesystem>


/*************************************************************************
Generating a label map
Author : Anupama Rajkumar
Date : 27.05.2020
Modified by: Jun Xiang 22,06,2020
Description : Idea is to create a single label map from a list of various
label classes.This map serves as points of reference when trying to classify
patches
**************************************************************************
*/
cv::Mat Utils::generateLabelMap(const std::vector<cv::Mat> & masks) {
	size_t NUMOFCLASSES = masks.size();
	int rows = masks[0].rows;
	int cols = masks[0].cols;
	cv::Mat labelMap = cv::Mat::zeros(rows, cols, CV_8UC1);
	for (size_t cnt = 0; cnt < NUMOFCLASSES; cnt++) {
		cv::Mat mask = masks[cnt];
		for (int row = 0; row < rows; ++row) {
			for (int col = 0; col < cols; ++col) {
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


/*
************************************************************************
input: the label 
return: the color
************************************************************************
*/
cv::Vec3b Utils::getLabelColor(unsigned char class_result)
{
	cv::Vec3b labelColor;

	// Color is BGR not RGB!
	cv::Vec3b black = cv::Vec3b(0, 0, 0);// unclassified, class 0

	cv::Vec3b red = cv::Vec3b(49, 60, 224); //city, class 1

	cv::Vec3b yellow = cv::Vec3b(0, 190, 246); //field, class 2

	cv::Vec3b dark_green = cv::Vec3b(66, 121, 79); //forest, class 3

	cv::Vec3b light_green = cv::Vec3b(0, 189, 181); // grassland, class 4

	cv::Vec3b blue = cv::Vec3b(164, 85, 50); //street, class 5

	std::vector<cv::Vec3b> right_color = { black, red,  yellow, dark_green, light_green,blue,};
	
	labelColor = right_color[int(class_result)];
	 
	return labelColor;
}


/*===================================================================
 * Function: generateColorMap
 *
 * Summary:
 *   Generate the colormap of classified results
 *
 * Arguments:
 *   std::string& hdf5_fileName - hdf5 filename
 *   std::string& feature_name - choose from { "texture", "color", "ctelements","polstatistic","decomp", "mp"}
 *   std::string & classifier_type - choose from {"KNN","opencvKNN", "opencvRF", "opencvFLANN"}
 *   int filterSize  
 *	 int patchSize   
 *	 int batchSize  
 * Returns:
 *   void
=====================================================================
*/
void Utils::generateColorMap(const std::string& hdf5_fileName, const std::string& feature_name, const std::string & classifier_type, int filterSize, int patchSize,int batchSize) {
	std::string parent = "/" + feature_name + "_filterSize_" + std::to_string(filterSize) + "_patchSize_" + std::to_string(patchSize);

	int totalrows = hdf5::getRowSize(hdf5_fileName, parent, "/" + classifier_type);

	std::vector<unsigned char> labels;
	cv::Mat labelMap;
	hdf5::readData(hdf5_fileName, "/masks", "/labelMap", labelMap);
	if(totalrows>0){
		std::vector<unsigned char> class_results;
		std::vector<unsigned char> ground_truth_labels;
		cv::Mat colorResultMap = cv::Mat::zeros(cv::Size(labelMap.size()), CV_8UC3);
		cv::Mat groundTruthMap = cv::Mat::zeros(cv::Size(labelMap.size()), CV_8UC3);

		int offset_row = 0;
		int partSize;
		if (batchSize > totalrows) { batchSize = totalrows; }
		int partsCount = totalrows / batchSize;
		for (int i = 0; i < partsCount; ++i) {
			partSize = totalrows / (partsCount - i);
			totalrows -= partSize;
			if (totalrows < 0) { break; }
			cv::Mat pts;
			hdf5::readData(hdf5_fileName, parent, "/" + classifier_type, pts, offset_row, partSize);

			for (int j = 0; j < pts.rows; j++) {
				int row = pts.at<int>(j, 1);
				int col = pts.at<int>(j, 2);
				unsigned char label = unsigned char(pts.at<int>(j, 0));
				unsigned char ground_truth = labelMap.at<unsigned char>(row, col);

				colorResultMap.at<cv::Vec3b>(row, col) = getLabelColor(label);
				groundTruthMap.at<cv::Vec3b>(row, col) = getLabelColor(ground_truth);

				class_results.push_back(label);
				ground_truth_labels.push_back(ground_truth);
			}
			offset_row = offset_row + partSize;
		}

		std::cout << std::endl;
		std::cout << "classifier: " << classifier_type << std::endl;
		std::cout << "generate " << feature_name + "_classresult.png" << std::endl;

		cv::imwrite(feature_name + "_colormap.png", colorResultMap);
		cv::imwrite("groundTruthMap.png", groundTruthMap);

		std::map<unsigned char, std::string> className = Utils::getClassName(hdf5_fileName);
		DataProcess::calculatePredictionAccuracy(feature_name, class_results, ground_truth_labels, className);
	}
	else {
		std::cout << "can't find " << parent + "/" + classifier_type << " in " << hdf5_fileName << std::endl;
	}
}

/*===================================================================
 * Function: generateFeatureMap
 *
 * Summary:
 *   Get the visulization of feature map for each single feature in feature group
 *
 * Arguments:
 *   std::string& hdf5_fileName - hdf5 filename
 *   std::string& feature_name - choose from { "texture", "color", "ctelements","polstatistic","decomp","mp"}
 *   int filterSize
 *	 int patchSize
 *	 int batchSize
 * Returns:
 *   void
=====================================================================
*/
void Utils::generateFeatureMap(const std::string& hdf5_fileName, const std::string& feature_name, int filterSize, int patchSize, int batchSize) {
	std::string parent = "/" + feature_name + "_filterSize_" + std::to_string(filterSize) + "_patchSize_" + std::to_string(patchSize);
	std::vector<std::string> dataset_name = {"/labelMap", "/groundtruth", "/feature" };

	//create folder for feature map images
	std::string outputpath = "featureMap_" + feature_name;
	std::filesystem::create_directories(outputpath);
	
	std::vector<int> cols; 
	std::vector<std::string> png_name;
	if (feature_name == "polstatistic") { 
		std::cout << "generate the feature map for the mean value of each polarimetric parameter" << std::endl;
		// the median value for each polarimetric parameter
		cols = {0,5,10,15,20,25,30,35,40,45 }; 
		png_name = {"Intensity_of_HH_channel","Intensity_of_HV_channel","Intensity_of_VV_channel","Phase_difference_HH-VV",
			"Co-polarize_ratio","Cross-polarized_ratio","HV_VV_ratio","Copolarization_ratio","Depolarization_ratio","Amplitude_of_HH-VV_correlation"};
	}
	else if (feature_name == "ctelements") {
		std::cout << "generate the feature map for each upper conner elements from C /T matrix" << std::endl;
		cols = { 4,13,22,31,40,49,58,67,76,85,94,103 };
		png_name = {  "T00","T01","T02","T11","T12","T22" ,"C00","C01","C02", "C11", "C12", "C22"};
	}
	else if (feature_name == "decomp") {
		std::cout << "generate the feature map for target decomposition component" << std::endl;
		//cols = { 4,13,22,31,40,49,58,67,76,85,94,103,112,121 };
		cols = { 4,13,22,31,40,76,85,94,103,112,121,130,139,148};
		png_name = { "cloude_entropy", "cloude_anisotropy", "cloude_alpha1" ,"cloude_alpha2","cloude_alpha3","freeman_surface","freeman_double-bounce" ,"freeman_volume","krogager_sphere", "krogager_diplane","krogager_helix","pauli_alpha" ,"pauli_beta", "pauli_gamma" };
	}
	else if (feature_name == "color") {
		std::cout << "generate the feature map for MPEG-7 DCD and CSD" << std::endl;
		cols = {0,1, 3, 32,33,34,35 };
		png_name = { "csd value 1","csd value 2","csd value 3","dominant_color_value1","dominant_color_value2","dominant_color_value3" ,"dominant_color_weight"};
	}
	else if (feature_name == "texture") {
		std::cout << "generate the feature map for texture on HH channel" << std::endl;
		cols = {0,1,8,9,16,17,24,25,32,33};
		png_name = { "GLCM engergy value 1", "GLCM engergy value 2","GLCM Contrast value 1","GLCM Contrast value 2","GLCM Homogenity value 1","GLCM Homogenity value 2","GLCM Entropy value 1","GLCM Entropy value 2","LBP value 1","LBP value 2" };
	}
	else if (feature_name == "mp") {
		std::cout << "generate the feature map for mp features on HH channel" << std::endl;
		png_name = { "opening","opening_by_construction","closing","closing_by_reconstruction"};
		cv::Mat hh;
		hdf5::readData(hdf5_fileName, "/masks", "/hh_intensity",hh);
		std::vector<cv::Mat> temp = morph::CaculateMP(hh, 3);

		for (int i = 0; i < temp.size(); i++) {
			temp[i].convertTo(temp[i], CV_8UC1);
			cv::equalizeHist(temp[i], temp[i]);
			cv::applyColorMap(temp[i], temp[i], cv::COLORMAP_JET);
			std::string outputpng = outputpath + "\\" + png_name[i] + ".png";
			std::cout << "generate " << outputpng << std::endl;
			cv::imwrite(outputpng, temp[i]);
		}
	}

	if( feature_name != "mp"){
		std::vector<cv::Mat> featureMap(png_name.size());

		cv::Mat labelMap;
		hdf5::readData(hdf5_fileName, "/masks", dataset_name[0], labelMap);
		for (auto& f : featureMap) { f = cv::Mat(cv::Size(labelMap.size()), CV_32FC1); }

		int totalrows = hdf5::getRowSize(hdf5_fileName, parent, "/feature");
		if (totalrows > 0) {
			int offset_row = 0;
			int partSize;
			if (batchSize > totalrows) { batchSize = totalrows; }
			int partsCount = totalrows / batchSize;
			for (int i = 0; i < partsCount; ++i) {
				partSize = totalrows / (partsCount - i);
				totalrows -= partSize;
				if (totalrows < 0) { break; }
				cv::Mat pts, feature;
				hdf5::readData(hdf5_fileName, parent, dataset_name[1], pts, offset_row, partSize);
				hdf5::readData(hdf5_fileName, parent, dataset_name[2], feature, offset_row, partSize);
				feature.convertTo(feature, CV_32FC1);

				for (int j = 0; j < feature.rows; j++) {
					int row = pts.at<int>(j, 1);
					int col = pts.at<int>(j, 2);
					for (int k = 0; k < cols.size(); k++) {
						featureMap[k].at<float>(row, col) = feature.at<float>(j, cols[k]);
					}
				}
				offset_row = offset_row + partSize;
			}

			int i = 0;

			for (auto& f : featureMap) {
				//get min and max, stretch it to 0-255
				//double mean = cv::mean(f)[0];
				// f = f - mean;
				//double min = 0, max = 0;
				//cv::minMaxLoc(f, &min, &max);
				//f = (f - min) * 255.0 / (max - min);
				f.convertTo(f, CV_8UC1);
				cv::equalizeHist(f, f);
				cv::applyColorMap(f, f, cv::COLORMAP_JET);
				std::string outputpng = outputpath + "\\" + png_name[i] + ".png";
				std::cout << "generate " << outputpng << std::endl;
				cv::imwrite(outputpng, f);
				i = i + 1;
			}
		}
		else {
			std::cout << "can't find " << parent + "/feature" << " in " << hdf5_fileName << std::endl;
		}
	}
}

/*===================================================================
 * Function: featureDimReduction
 *
 * Summary:
 *   reduced the feature dimension by T-SNE
 *	 dump the first batch to txt file for plotting
 *	 check the KNN accuracy on reduced feature data
 *
 * Arguments:
 *   std::string& hdf5_fileName - hdf5 filename
 *   std::string& feature_name - choose from { "texture", "color", "ctelements","polstatistic","decomp", "mp"}
 *   int filterSize
 *	 int patchSize
 *	 int batchSize
 * Returns:
 *   void
=====================================================================
*/
void Utils::featureDimReduction(const std::string& hdf5_fileName, const std::string& feature_name, int filterSize, int patchSize, int batchSize) {

	std::vector<std::string> dataset_name = { "/feature" ,"/groundtruth","/dimReduced_feature" };
	std::string parent = "/" + feature_name + "_filterSize_" + std::to_string(filterSize) + "_patchSize_" + std::to_string(patchSize);
	cv::Mat  feature, groundtruths;
	hdf5::readData(hdf5_fileName, parent, dataset_name[0], feature, 0, batchSize);
	hdf5::readData(hdf5_fileName, parent, dataset_name[1], groundtruths, 0, batchSize);
	std::cout << "get " << feature.rows << " rows for " << feature_name << " feature" << std::endl;

	cv::Mat reduced_feature = DataProcess::featureDimReduction(feature, 2);

	std::string dim_reduce = "dimReduced_" + feature_name + ".txt";
	std::cout << "save dimension reduced feature to " << dim_reduce << std::endl;
	std::ofstream fout(dim_reduce);

	std::vector<cv::Mat> newfeatures(reduced_feature.rows);
	std::vector<unsigned char> labels(reduced_feature.rows);
	for (int i = 0; i < reduced_feature.rows; i++) {
		//dump this batch to txt file for ploting
		fout << groundtruths.at<int>(i, 0) << "," << reduced_feature.at<float>(i, 0) << "," << reduced_feature.at<float>(i, 1) << std::endl;

		cv::Mat temp(1, 2, CV_32FC1);
		temp.at<float>(0, 0) = reduced_feature.at<float>(i, 0);
		temp.at<float>(0, 1) = reduced_feature.at<float>(i, 1);
		newfeatures[i] = temp;
		labels[i] = groundtruths.at<int>(i, 0);
	}
	std::vector<unsigned char> results;
	//check the KNN accuracy of dim reduced features
	DataProcess::applyML(newfeatures, labels, 80, "opencvFLANN", results);
	std::map<unsigned char, std::string> className = Utils::getClassName(hdf5_fileName);
	DataProcess::calculatePredictionAccuracy("", results, labels, className);
}

/*===================================================================
 * Function: classifyFeaturesML
 *
 * Summary:
 *   Read the features from hdf5 file, classify them and write the classifiy results into hdf5 file
 *
 * Arguments:
 *   std::string& hdf5_fileName - hdf5 filename
 *   std::string& feature_name - choose from { "texture", "color", "ctelements","polstatistic","decomp", "mp"}
 *   std::string & classifier_type - choose from {"KNN","opencvKNN", "opencvRF", "opencvFLANN"}
 *	 int trainPercent
 *   int filterSize
 *	 int patchSize
 *	 int batchSize
 * Returns:
 *   void
=====================================================================
*/
void Utils::classifyFeaturesML(const std::string& hdf5_fileName, const std::string& feature_name, const std::string classifier_type, int trainPercent, int filterSize, int patchSize,int batchSize) {
	std::string parent = "/" + feature_name + "_filterSize_" + std::to_string(filterSize) + "_patchSize_" + std::to_string(patchSize);
	std::vector<std::string> dataset_name = { "/feature" ,"/groundtruth" };

	std::map<unsigned char, std::string> className = Utils::getClassName(hdf5_fileName);

	if (hdf5::checkExist(hdf5_fileName, parent, "/" + classifier_type)) {
		hdf5::deleteData(hdf5_fileName, parent, "/" + classifier_type );
	}

	int fullSize = hdf5::getRowSize(hdf5_fileName, parent, dataset_name[0]);
	std::cout << "get " << fullSize << " rows for " << feature_name << " feature from hdf5 file with filterSize " << filterSize << " , patchSize " << patchSize << std::endl;
	
	int offset_row = 0;
	int partSize;
	if (batchSize > fullSize) { batchSize = fullSize; }
	int partsCount = fullSize / batchSize;
	if (fullSize != 0) {
		for (int i = 0; i < partsCount; ++i) {
			partSize = fullSize / (partsCount - i);
			fullSize -= partSize;
			if (fullSize < 0) { break; }
			std::vector<cv::Mat> features;
			std::vector<cv::Point> labelPoints;
			std::vector<unsigned char> labels;
			Utils::getFeaturesFromHDF(hdf5_fileName, parent, features, labels, labelPoints, offset_row, partSize);
			std::cout << "get " << features.size() << " rows for " << feature_name << " feature from hdf5 file with filterSize " << filterSize << " , patchSize " << patchSize << std::endl;

			std::vector<unsigned char> class_results;
			DataProcess::applyML(features, labels, 80, classifier_type, class_results);
			DataProcess::calculatePredictionAccuracy("", class_results, labels, className);
			saveClassResultToHDF(hdf5_fileName, parent, classifier_type, class_results, labelPoints);
			offset_row = offset_row + partSize;
			std::cout << "classifiy " << feature_name << " progress: " << float(i + 1) / float(partsCount) * 100.0 << "% \n" << std::endl;
		}
	}
	else {
		std::cout << feature_name << " with filterSize " << filterSize << " , patchSize " << patchSize << " is not existed in hdf5 file " << std::endl;
	}
}


/*===================================================================
 * Function: saveClassResultToHDF
 *
 * Summary:
 *   write back the classified results to hdf5 ( class result from classifier,sample points)
 *
 * Arguments:
 *   std::string& hdf5_fileName - hdf5 filename
 *   std::string& parent_name
 *   std::string & classResult_name - choose from {"KNN","opencvKNN", "opencvRF", "opencvFLANN"}
 *	 std::vector<unsigned char>& class_result
 *	 std::vector<cv::Point> & points
 * output:
 *  
=====================================================================
*/
void Utils::saveClassResultToHDF(const std::string& hdf5_fileName, const std::string& parent_name, const std::string& classResult_name, const std::vector<unsigned char>& class_result, const std::vector<cv::Point> & points) {
	cv::Mat pts = cv::Mat(points.size(), 3, CV_32SC1);
	for (size_t i = 0; i < points.size(); ++i) {
		pts.at<int>(i, 0) = (int)(class_result[i]);
		pts.at<int>(i, 1) = points[i].y; //row
		pts.at<int>(i, 2) = points[i].x; //col
	}
	hdf5::insertData(hdf5_fileName, parent_name, "/"+classResult_name, pts);
}


/*===================================================================
 * Function: getFeaturesFromHDF
 *
 * Summary:
 *   get features data and its groundtruth from hdf5
 *
 * Arguments:
 *   std::string& hdf5_fileName - hdf5 filename
 *   std::string& parent_name  
 *	 int batchSize
 *	 int offset_row - the start row
 *   int counts_rows - the number of samples for output
 * output:
 *	 std::vector<cv::Mat>& features 
 *   std::vector<unsigned char>& featureLabels 
 *	 std::vector<cv::Point> & labelPoints 
=====================================================================
*/
void Utils::getFeaturesFromHDF(const std::string& hdf5_fileName, const std::string& parent_name,std::vector<cv::Mat>& features,std::vector<unsigned char>& featureLabels, std::vector<cv::Point> & labelPoints, int offset_row, int counts_rows) {
	std::vector<std::string> dataset_name = { "/feature" ,"/groundtruth" };
	cv::Mat feature, pts;
	if (hdf5::checkExist(hdf5_fileName, parent_name, dataset_name[0]) &&
		hdf5::checkExist(hdf5_fileName, parent_name, dataset_name[1])) {
		hdf5::readData(hdf5_fileName, parent_name, dataset_name[0], feature, offset_row,counts_rows);
		hdf5::readData(hdf5_fileName, parent_name, dataset_name[1], pts, offset_row, counts_rows);

		for (int i = 0; i < feature.rows; ++i) {
			features.push_back(feature.row(i));
			featureLabels.push_back((unsigned char)(pts.at<int>(i, 0)));
			cv::Point p;
			p.y = pts.at<int>(i, 1); //row
			p.x = pts.at<int>(i, 2); //col
			labelPoints.push_back(p);
		}
	}
}


/*===================================================================
 * Function: generateSamplePoints
 *
 * Summary:
 *   generate all the possible sample points
 *
 * Arguments:
 *   cv::Mat& labelMap  
 *   int filterSize
 *	 int patchSize
 *	 int& stride
 * output:
 *	 std::vector<cv::Point> samplePoints
=====================================================================
*/
std::vector<cv::Point> Utils::generateSamplePoints(const cv::Mat& labelMap, const int& patchSize, const int& stride) {

	std::vector<cv::Point> samplePoints;
	for (int row = 0; row < labelMap.rows - patchSize; row += stride) {
		for (int col = 0; col < labelMap.cols - patchSize; col += stride) {
			cv::Rect cell = cv::Rect(col, row, patchSize, patchSize);

			int halfsize = patchSize / 2;

			//record the central points of each patch
			samplePoints.push_back(cv::Point(col + halfsize, row + halfsize));
		}
	}
	return samplePoints;
}


/*===================================================================
 * Function: getRandomSamplePoint
 *
 * Summary:
 *   get random samples of homogeneous area for one type of class
 *
 * Arguments:
 *   cv::Mat& labelMap
 *	 unsigned char& sampleLabel
 *   int filterSize
 *	 int sampleSize
 *	 int& stride
 *   int& numOfSamplePointPerClass - 0 means to return all the possible sample points
 * output:
 *	 std::vector<cv::Point>& samplePoints
=====================================================================
*/
void Utils::getRandomSamplePoint(const cv::Mat& labelMap, std::vector<cv::Point>& samplePoints, const unsigned char& sampleLabel, const int& sampleSize, const int& stride, const int& numOfSamplePointPerClass) {

	std::vector<cv::Point> temp = generateSamplePoints(labelMap, sampleSize, stride);
	std::map<unsigned char, std::vector<cv::Point> > count;
	for (auto& p : temp) {
		unsigned char label = labelMap.at<unsigned char>(p.y, p.x);
		if (label == sampleLabel) {
			count[sampleLabel].push_back(p);
		}
	}
	std::vector<cv::Point> pts = count[sampleLabel];

	if (numOfSamplePointPerClass > 0) {
		std::random_device random_device;
		std::mt19937 engine{ random_device() };
		std::uniform_int_distribution<int> pt(0, pts.size() - 1);
		size_t num = 0;
		size_t iter = 0;

		while (num < numOfSamplePointPerClass) {
			cv::Point p = pts[pt(engine)];

			// get samples in homogeneous areas 
			// this is only for checking the accuracy of features
			unsigned char label = labelMap.at<unsigned char>(p.y, p.x);
			unsigned char sample_upcorner = labelMap.at<unsigned char>(p.y - sampleSize / 2, p.x - sampleSize / 2);
			unsigned char sample_downcorner = labelMap.at<unsigned char>(p.y + sampleSize / 2, p.x + sampleSize / 2);
			unsigned char sample_leftcorner = labelMap.at<unsigned char>(p.y + sampleSize / 2, p.x - sampleSize / 2);
			unsigned char sample_rightcorner = labelMap.at<unsigned char>(p.y - sampleSize / 2, p.x + sampleSize / 2);
			if ((label == sample_upcorner) && (label == sample_downcorner) &&
				(label == sample_leftcorner) && (label == sample_rightcorner)) {
				samplePoints.push_back(p);
				++num;
			}
			++iter;
			if (iter > pts.size()) { break; }
		}
	}
	else {
		std::cout << "load all the sample points" << std::endl;
		copy(pts.begin(), pts.end(), back_inserter(samplePoints));
	}
}


/*===================================================================
 * Function: splitVec
 *
 * Summary:
 *   split index to batches, make sure the distribution of each class in each batch is the same as it in the whole data
 *	 shuffle the index
 *
 * Arguments:
 *   std::vector<unsigned char>& labels
 *   int batchSize
 * output:
 *	 std::vector<std::vector<int>>& subInd
=====================================================================
*/
void Utils::splitVec(const std::vector<unsigned char>& labels, std::vector<std::vector<int>>& subInd, int batchSize) {

	// To regulate count of parts
	if (batchSize > labels.size()) { batchSize = labels.size(); }
	int partsCount = labels.size() / batchSize;

	if (subInd.size() == 0) { subInd = std::vector<std::vector<int>>(partsCount); }

	std::map<unsigned char, std::vector<int>> count;
	for (int ind = 0; ind < labels.size(); ind++) {
		count[labels[ind]].push_back(ind);
	}

	for (const auto& c : count) {
		std::vector<int> inds = c.second;
		// Variable to control size of non divided elements
		int fullSize = inds.size();
		int start = 0;
		for (int i = 0; i < partsCount; ++i) {
			int partSize = fullSize / (partsCount - i);
			fullSize -= partSize;
			for (int j = 0; j < partSize; j++) {
				subInd[i].push_back(inds[start + j]);
			}
			start = start + partSize;
		}
	}

	// shuffle the index
	// obtain a time-based seed
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	std::default_random_engine e(seed);
	for (auto i = 0; i < subInd.size(); i++) {
		std::shuffle(subInd[i].begin(), subInd[i].end(), e);
	}
}


///*===================================================================
// * Function: featureDimReduction
// *
// * Summary:
// *   reduced the feature dimension by T-SNE
// *	 dump the first batch to txt file for plotting
// *	 check the KNN accuracy on reduced feature data
// *
// * Arguments:
// *   std::string& hdf5_fileName - hdf5 filename
// *   std::string& feature_name - choose from { "texture", "color", "ctelements","polstatistic","decomp", "mp"}
// *   int filterSize
// *	 int patchSize
// *	 int batchSize
// * Returns:
// *   void
//=====================================================================
//*/
//void Utils::featureDimReduction(const std::string& hdf5_fileName, const std::string& feature_name, int filterSize, int patchSize,int batchSize) {
//	
//	std::vector<std::string> dataset_name = { "/feature" ,"/groundtruth" };
//	std::string parent = "/" + feature_name + "_filterSize_" + std::to_string(filterSize) + "_patchSize_" + std::to_string(patchSize);
//	int fullSize = hdf5::getRowSize(hdf5_fileName, parent, dataset_name[0]);
//	std::cout << "get " << fullSize << " rows for " << feature_name << " feature from hdf5 file with filterSize " << filterSize << " , patchSize " << patchSize << std::endl;
//
//	int offset_row = 0;
//	int partSize;
//	if (batchSize > fullSize) { batchSize = fullSize; }
//	int partsCount = fullSize / batchSize;
//	cv::Mat  feature, groundtruths, reduced_feature;
//
//	if (fullSize != 0) {
//		for (int i = 0; i < partsCount; ++i) {
//			partSize = fullSize / (partsCount - i);
//			fullSize -= partSize;
//			if (fullSize < 0) { break; }
//			hdf5::readData(hdf5_fileName, parent, dataset_name[0], feature, offset_row, partSize);
//			std::cout << "get " << feature.rows << " rows for " << feature_name << " feature" << std::endl;
//			reduced_feature = DataProcess::featureDimReduction(feature, 3);
//			hdf5::insertData(hdf5_fileName, parent, "/feature_dimReduced", reduced_feature);
//
//			offset_row = offset_row + partSize;
//			std::cout << "classifiy " << feature_name << " progress: " << float(i + 1) / float(partsCount) * 100.0 << "% \n" << std::endl;
//		}
//	}
//	else {
//		std::cout << feature_name << " with filterSize " << filterSize << " , patchSize " << patchSize << " is not existed in hdf5 file " << std::endl;
//	}
//}


 std::map<unsigned char, std::string> Utils::getClassName(const std::string& filename) {
	 // get the class names from hdf5
	 cv::Mat classlabels;
	 hdf5::readAttr(filename, "classlabels", classlabels);
	 std::map<unsigned char, std::string> className;
	 for(int i=0; i< classlabels.cols; i++) {
		 unsigned char label = classlabels.at<unsigned char>(0, i);
		 std::string class_name;
		 hdf5::readAttr(filename, std::to_string(label), class_name);
		 if (!class_name.empty()) { className[label] = class_name;}
	 }
	 return className;
 }



