#include "DataProcess.hpp"

/*===================================================================
 * Function: DivideTrainTestData
 *
 * Summary:
 *   split the data into train/test set balancely in different classes
 *	 return the index of the test data in original data
 *
 * Arguments:
 *   const std::vector<cv::Mat>& data
 *   const std::vector<unsigned char>& data_label
 *	 int percentOfTrain
 *	 int fold - crossvalidation number,an integer between {1, 100 / (100 - percentOfTrain)}
 * output:
 *	std::vector<cv::Mat>& train_img
 *	std::vector<unsigned char>& train_label
 *	std::vector<cv::Mat>& test_img
 *	std::vector<unsigned char>& test_label
=====================================================================
*/
std::vector<int> DataProcess::DivideTrainTestData(const std::vector<cv::Mat>& data, const std::vector<unsigned char>& data_label, int percentOfTrain,
	std::vector<cv::Mat>& train_img, std::vector<unsigned char>& train_label, std::vector<cv::Mat>& test_img, std::vector<unsigned char>& test_label, int fold) {

	std::map<unsigned char, std::vector<int>> numPerClass;
	int index = 0;
	for (auto c : data_label) {
		numPerClass[c].push_back(index);
		index++;
	}
	std::vector<int> train_index, test_index;
	int total_folds = 100 / (100 - percentOfTrain);

	// make sure each class is divided 80% as train, 20% as test
	for (auto it = numPerClass.begin(); it != numPerClass.end(); it++)
	{
		size_t test_size = it->second.size() *(100 - percentOfTrain )/ 100;
		size_t train_size = it->second.size() - test_size;
		if( train_size >0 && test_size >0){
			std::vector<int> indOfClass;
			// expand indOfClass twice
			copy(it->second.begin(), it->second.end(), back_inserter(indOfClass));
			copy(it->second.begin(), it->second.end(), back_inserter(indOfClass));

			std::vector<int> train_temp, test_temp;
			int train_temp_size = 0;
			int test_temp_size = 0;
			for (size_t i = 0; i < indOfClass.size(); ++i) {
				if (train_temp_size < test_size) {
					test_temp.push_back(indOfClass[i + (fold - 1) * test_size]);
					train_temp_size++;
				}
				if (test_temp_size < train_size) {
					train_temp.push_back(indOfClass[i + fold * test_size]);
					test_temp_size++;
				}
			}
			copy(train_temp.begin(), train_temp.end(), back_inserter(train_index));
			copy(test_temp.begin(), test_temp.end(), back_inserter(test_index));
		}
	}
	for (auto i : train_index) {
		train_img.push_back(data[i]);
		train_label.push_back(data_label[i]);
	}
	for (auto i : test_index) {
		test_img.push_back(data[i]);
		test_label.push_back(data_label[i]);
	}

	return test_index;
}


/*===================================================================
 * Function: shuffleDataSet
 *
 * Summary:
 *   shuffle the data, and return the original index of the shuffled data
 *
 * Arguments:
 *   std::vector<cv::Mat>& data  
 *   std::vector<unsigned char>& data_label
 * output:
 *	std::vector<int> original index of the shuffled data
=====================================================================
*/
std::vector<int> DataProcess::shuffleDataSet(std::vector<cv::Mat>& data, std::vector<unsigned char>& data_label) {
	int size = data.size();
	std::vector<int> ind(size);
	std::random_device random_device;
	std::mt19937 engine{ random_device() };
	std::uniform_int_distribution<int> rnd(0, size - 1);
	for (int i = 0; i < size; ++i) {
		cv::Mat temp = data[i];
		unsigned char temp_c = data_label[i];
		int swap = rnd(engine);
		if (swap == i) { continue; }
		else {
			data[i] = data[swap];
			data[swap] = temp;
			data_label[i] = data_label[swap];
			data_label[swap] = temp_c;
		}
		ind[i] = swap;
		ind[swap] = i;
	}
	return ind;
}

/*===================================================================
 * Function: calculatePredictionAccuracy
 *
 * Summary:
 *   calculate the accuracy for each class, and return the overal accuracy
 *	 if provided feature_name, write accuracy to txt file
 *
 * Arguments:
 *   const std::string& feature_name -  choose from { "/texture", "/color", "/ctElements","/polStatistic","/decomp", "/MP"}
 *   const std::vector<unsigned char>& classResult 
 *	 const std::vector<unsigned char>& groundtruth 
 *	 const std::map<unsigned char, std::string>& className - class name for each class label
 * output:
 *	float accuracy
=====================================================================
*/
float DataProcess::calculatePredictionAccuracy(const std::string& feature_name, const std::vector<unsigned char>& classResult, const std::vector<unsigned char>& groundtruth, const std::map<unsigned char, std::string>& className)
{
	std::string overall_accuracy;
	std::ofstream fout;
	if (!feature_name.empty()) { 
		overall_accuracy = "oa_" + feature_name+ ".txt";
		fout.open(overall_accuracy);
	}
	
	float accuracy = 0.0;
	if (classResult.size() != groundtruth.size()) {
		std::cerr << "Predicted and actual label vectors differ in length. Something doesn't seem right." << std::endl;
		exit(-1);
	}
	else {
		std::map<unsigned char, float> hit;
		std::map<unsigned char, float> total;

		int dim = classResult.size();

		for (int i = 0; i < dim; ++i) {
			if ((classResult[i] == groundtruth[i])&& (groundtruth[i] != unsigned char(0))) {
				hit[classResult[i]]++;
			}
			total[groundtruth[i]]++;
		}

		float a = 0.0;
		for (auto& h : hit) {
			unsigned char label = h.first;
			std::string classname = className.at(label);
			float correct = h.second;
			float totalNum = total[label];
			float class_accuracy = correct / totalNum;
			a = correct + a;
			std::cout << "accuracy for class " << classname << ": " << class_accuracy << std::endl;

			if (!feature_name.empty()) {
				fout << "accuracy for class " << classname << ": " << class_accuracy << std::endl;
			}
		}
		accuracy = a / groundtruth.size();
		std::cout << "overall accuracy: " << accuracy << std::endl;

		if (!feature_name.empty()) {
			fout << "oa: " << accuracy << std::endl;
		}
	}
	return  accuracy;
}

/*===================================================================
 * Function: getConfusionMatrix
 *
 * Summary:
 *   calculate the ConfusionMatrix from class results and groundtruth
 *
 * Arguments:
 *   const std::string& feature_name -  choose from { "/texture", "/color", "/ctElements","/polStatistic","/decomp", "/MP"}
 *   const std::vector<unsigned char>& classResult
 *	 const std::vector<unsigned char>& groundtruth
 * output:
 *	cv::Mat 
=====================================================================
*/
cv::Mat DataProcess::getConfusionMatrix(const std::map<unsigned char, std::string>& className, std::vector<unsigned char>& classResult, std::vector<unsigned char>& groundtruth) {
	std::map<std::pair<unsigned char, signed char>, int> testCount;

	for (int i = 0; i < groundtruth.size(); ++i) {
		for (int j = 0; j < classResult.size(); ++j) {
			std::pair temp = std::make_pair(groundtruth[i], classResult[j]);
			testCount[temp]++;
		}
	}

	int numOfClass = className.size();
	std::vector<unsigned char> classList(numOfClass);
	for (auto it = className.begin(); it != className.end(); it++) {
		classList.push_back(it->first);
	}

	cv::Mat count = cv::Mat(className.size(), className.size(), CV_8UC1);
	for (int i = 0; i < numOfClass; ++i) {
		for (int j = 0; j < numOfClass; ++j) {
			std::pair temp = std::make_pair(classList[i], classList[j]);
			count.at<unsigned char>(i, j) = testCount[temp];
		}
	}
	return count;
}

/*===================================================================
 * Function: applyML
 *
 * Summary:
 *   classify the data, run cross validation on each test part, get class results
 *
 * Arguments:
 *   const std::vector<cv::Mat>& data - original data
 *   const std::vector<unsigned char>& data_labels  
 *	 int trainPercent
 *   const std::string& classifier_type - choose from {"KNN","opencvKNN", "opencvRF", "opencvFLANN"}
 *   const std::vector<unsigned char>& classResult
 *	 int K - the number for KNN, opencvKNN, opencvRF
 * output:
 *	std::vector<unsigned char>& results
=====================================================================
*/
void DataProcess::applyML(const std::vector<cv::Mat>& data, const std::vector<unsigned char>& data_labels, int trainPercent, const std::string& classifier_type, std::vector<unsigned char>& results,int K) {

	std::cout << "start to classify data with classifier :" << classifier_type << std::endl;
	std::cout << "data size :" << data.size() << std::endl;
	// classify result
	results = std::vector<unsigned char>(data_labels.size());

	//copy the original data
	std::vector<cv::Mat> temp(data.begin(), data.end());
	std::vector<unsigned char> temp_labels(data_labels.begin(), data_labels.end());

	std::vector<cv::Mat> train;
	std::vector<unsigned char> train_labels;
	std::vector<cv::Mat> test;
	std::vector<unsigned char> test_labels;

	int total_folds = 100 / (100 - trainPercent);
	for (int fold = 1; fold < total_folds + 1; ++fold) {
		std::vector<int> test_ind = DataProcess::DivideTrainTestData(temp, temp_labels, trainPercent, train, train_labels, test, test_labels, fold);
		std::vector<unsigned char> test_result;
		cv::Mat traindata, traindata_label, testdata;
		vconcat(train, traindata);
		vconcat(test, testdata);
		vconcat(train_labels, traindata_label);
		traindata_label.convertTo(traindata_label, CV_32SC1);
		traindata.convertTo(traindata, CV_32FC1);
		testdata.convertTo(testdata, CV_32FC1);

		if (classifier_type == "opencvKNN") {
			cv::Ptr<cv::ml::TrainData> cv_data = cv::ml::TrainData::create(traindata, 0, traindata_label);
			cv::Ptr<cv::ml::KNearest>  knn(cv::ml::KNearest::create());
			knn->setDefaultK(K);
			knn->setIsClassifier(true);
			knn->train(cv_data);
			for (auto& x_test : test) {
				x_test.convertTo(x_test, CV_32FC1);
				auto knn_result = knn->predict(x_test);
				test_result.push_back(unsigned char(knn_result));
			}
		}
		else if (classifier_type == "KNN") {
			KNN* knn = new KNN();
			knn->KNNTest(train, train_labels, test, test_labels, K, test_result);
			delete knn;
		}
		else if (classifier_type == "opencvFLANN") {
			cv::flann::Index flann_index(
				traindata,
				cv::flann::KDTreeIndexParams(4),
				cvflann::FLANN_DIST_EUCLIDEAN
			);
			cv::Mat indices(testdata.rows, K, CV_32S);
			cv::Mat dists(testdata.rows, K, CV_32F);
			flann_index.knnSearch(testdata, indices, dists, K, cv::flann::SearchParams(200));
			KNN* knn = new KNN();
			for (int i = 0; i < testdata.rows; i++) {
				std::vector<std::pair<float, unsigned char>> dist_vec(K);
				for (int j = 0; j < K; j++) {
					unsigned char temp = train_labels[indices.at<int>(i, j)];
					float distance = dists.at<float>(i, j);
					dist_vec[j] = std::make_pair(distance, temp);
				}
				// voting 
				test_result.push_back(knn->Classify(dist_vec, K));
				dist_vec.clear();
			}
			delete knn;
		}
		else if (classifier_type == "opencvRF") {
			cv::Ptr<cv::ml::TrainData> cv_data = cv::ml::TrainData::create(traindata, cv::ml::ROW_SAMPLE, traindata_label);
			cv::Ptr<cv::ml::RTrees>  randomForest(cv::ml::RTrees::create());
			auto criterRamdomF = cv::TermCriteria();
			criterRamdomF.type = cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS;
			criterRamdomF.epsilon = 1e-8;
			criterRamdomF.maxCount = 500;
			randomForest->setTermCriteria(criterRamdomF);
			randomForest->setMaxCategories(K);
			randomForest->setMaxDepth(K);

			randomForest->train(cv_data);
			for (auto& x_test : test) {
				x_test.convertTo(x_test, CV_32FC1);
				test_result.push_back(unsigned char(randomForest->predict(x_test)));
			}
		}

		for (int i = 0; i < test_result.size(); ++i) {
			//results[original_ind[test_ind[i]]] = test_result[i];
			results[test_ind[i]] = test_result[i];
		}

		train.clear();
		train_labels.clear();
		test.clear();
		test_labels.clear();
		test_result.clear();
	}
}


/*===================================================================
 * Function: featureDimReduction
 *
 * Summary:
 *   descrease the dimension by t-sne
 *
 * Arguments:
 *   const cv::Mat& features - each row is a sample
 *	 int new_dims - the reduced dimension size
 * output:
 *	cv::Mat size( features.rows, new_dims)
=====================================================================
*/
cv::Mat DataProcess::featureDimReduction(const cv::Mat& features, int new_dims) {
	cv::Mat feature;
	features.convertTo(feature, CV_64FC1);
	// Define some variables
	int N = feature.rows;
	int D = feature.cols;
	int perplexity = 40;
	int max_iter = 1000;
	double* X = (double*)malloc(feature.total() * sizeof(double)); // data
	double* Y = (double*)malloc(N * new_dims * sizeof(double));//output

	//data
	for (int i = 0; i < feature.rows; i++) {
		for (int j = 0; j < feature.cols; j++) {
			X[i * feature.cols + j] = feature.at<double>(i, j);
		}
	}
	TSNE::run(X, N, D, Y, new_dims, perplexity, 0.5, -1, false, max_iter, 250, 250);


	cv::Mat  reduced_feature(N, new_dims, CV_32FC1);
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < new_dims; j++) {
			reduced_feature.at<float>(i, j) = float(Y[i * new_dims + j]);
		}
	}

	// Clean up the memory
	free(X); X = NULL;
	free(Y); Y = NULL;
	return reduced_feature;
}