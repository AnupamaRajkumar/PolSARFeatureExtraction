#include "dataset_hdf5.hpp"

void hdf5::deleteData(const std::string& filename, const std::string& parent_name, const std::string& dataset_name) {
	cv::Ptr<hdf::HDF5> h5io = hdf::open(filename);
	std::string datasetName = parent_name + dataset_name;

	if (!h5io->hlexists(parent_name)) {
		std::cout << parent_name << " is not existed." << std::endl;
	}
	else {
		if (!h5io->hlexists(datasetName)) {
			std::cout << datasetName << " is not existed." << std::endl;
		}
		else {
			int result = h5io->dsdelete(datasetName);
			if (!result) {
				std::cout << "delete dataset " << datasetName << " success." << std::endl;
			}
			else {
				std::cout << "Failed to delete " << datasetName << std::endl;
			}
		}
	}
}


void hdf5::writeData(const std::string& filename, const std::string& parent_name, const std::string& dataset_name, const cv::Mat& src) {
	if (!src.empty()) {
		cv::Mat data = src.clone();
		if (data.channels() > 1) {
			for (size_t i = 0; i < data.total() * data.channels(); ++i)
				((int*)data.data)[i] = (int)i;
		}

		cv::Ptr<hdf::HDF5> h5io = hdf::open(filename);

		// first we need to create the parent group
		if (!h5io->hlexists(parent_name)) h5io->grcreate(parent_name);

		// create the dataset if it not exists
		std::string datasetName = parent_name + dataset_name;
		if (!h5io->hlexists(datasetName)) {
			h5io->dscreate(data.rows, data.cols, data.type(), datasetName);
			h5io->dswrite(data, datasetName);

			// check if the data are correctly write to hdf file
			cv::Mat expected = cv::Mat(cv::Size(data.size()), data.type());
			h5io->dsread(expected, datasetName);
			float diff = norm(data - expected);
			CV_Assert(abs(diff) < 1e-10);

			if (h5io->hlexists(datasetName))
			{
				//std::cout << "write " << datasetName << " to " << filename << " success." << std::endl;
			}
			else {
				std::cout << "Failed to write " << datasetName << " to " << filename << std::endl;
			}
		}
		else {
			std::cout << datasetName << " is already existed." << std::endl;
		}
		h5io->close();
	}
}


void hdf5::readData(const std::string& filename, const std::string& parent_name, const std::string& dataset_name, cv::Mat& data, int offset_row, int counts_rows) {
	cv::Ptr<hdf::HDF5> h5io = hdf::open(filename);

	std::string datasetName = parent_name + dataset_name;

	if (!h5io->hlexists(parent_name)) {
		//std::cout << parent_name << " is not existed" << std::endl;
		data = cv::Mat();
	}
	else if (!h5io->hlexists(datasetName)) {
		//std::cout << datasetName << " is not existed" << std::endl;  
		data = cv::Mat();
	}
	else {

		std::vector<int> data_size = h5io->dsgetsize(datasetName);

		if (counts_rows != 0 && counts_rows <= data_size[0]) {
			data = cv::Mat(counts_rows, data_size[1], h5io->dsgettype(datasetName));
			std::vector<int> dims_offset(2), dims_count(2);
			dims_offset = { offset_row,0 };
			dims_count = { counts_rows, data.cols };
			h5io->dsread(data, datasetName, dims_offset, dims_count);
		}
		else if (counts_rows == 0) {
			data = cv::Mat(data_size[0], data_size[1], h5io->dsgettype(datasetName));

			h5io->dsread(data, datasetName);
		}
	}

	h5io->close();
}


int hdf5::getRowSize(const std::string& filename, const std::string& parent_name, const std::string& dataset_name) {
	int rows;

	std::vector<int> data_size;
	cv::Ptr<hdf::HDF5> h5io = hdf::open(filename);

	std::string datasetName = parent_name + dataset_name;

	if (!h5io->hlexists(parent_name)) {
		//std::cout << parent_name << " is not existed" << std::endl;
		rows = 0;
	}
	else if (!h5io->hlexists(datasetName)) {
		//std::cout << datasetName << " is not existed" << std::endl;  
		rows = 0;
	}
	else {

		data_size = h5io->dsgetsize(datasetName);
		rows = data_size[0];
	}

	h5io->close();
	return rows;
}

int hdf5::getColSize(const std::string& filename, const std::string& parent_name, const std::string& dataset_name) {
	int cols;

	std::vector<int> data_size;
	cv::Ptr<hdf::HDF5> h5io = hdf::open(filename);

	std::string datasetName = parent_name + dataset_name;

	if (!h5io->hlexists(parent_name)) {
		//std::cout << parent_name << " is not existed" << std::endl;
		cols = 0;
	}
	else if (!h5io->hlexists(datasetName)) {
		//std::cout << datasetName << " is not existed" << std::endl;  
		cols = 0;
	}
	else {

		data_size = h5io->dsgetsize(datasetName);
		cols = data_size[1];
	}

	h5io->close();
	return cols;
}

void hdf5::writeAttr(const std::string& filename, const std::string& attribute_name, const int& attribute_value) {
	cv::Ptr<hdf::HDF5> h5io = hdf::open(filename);
	if (!h5io->atexists(attribute_name)) {
		h5io->atwrite(attribute_value, attribute_name);
	}
	else {
		std::cout << " already existed" << std::endl;
	}
	h5io->close();
}


void hdf5::writeAttr(const std::string& filename, const std::string& attribute_name, const std::string& attribute_value) {
	cv::Ptr<hdf::HDF5> h5io = hdf::open(filename);
	if (!h5io->atexists(attribute_name)) {
		h5io->atwrite(attribute_value, attribute_name);
	}
	else {
		std::cout << " already existed" << std::endl;
	}
	h5io->close();
}

void hdf5::writeAttr(const std::string& filename, const std::string& attribute_name, const cv::Mat& attribute_value) {
	cv::Ptr<hdf::HDF5> h5io = hdf::open(filename);
	if (!h5io->atexists(attribute_name)) {
		h5io->atwrite(attribute_value, attribute_name);
	}
	else {
		std::cout << " already existed" << std::endl;
	}
	h5io->close();
}

void hdf5::readAttr(const std::string& filename, const std::string& attribute_name, std::string& attribute_value) {
	cv::Ptr<hdf::HDF5> h5io = hdf::open(filename);
	if (!h5io->atexists(attribute_name)) {
		std::cout << attribute_name << " is not existed" << std::endl;
	}
	else {
		h5io->atread(&attribute_value, attribute_name);
	}
	h5io->close();
}


void hdf5::readAttr(const std::string& filename, const std::string& attribute_name, int& attribute_value) {
	cv::Ptr<hdf::HDF5> h5io = hdf::open(filename);
	if (!h5io->atexists(attribute_name)) {
		std::cout << attribute_name << " is not existed" << std::endl;
	}
	else {
		h5io->atread(&attribute_value, attribute_name);
	}
	h5io->close();
}

void hdf5::readAttr(const std::string& filename, const std::string& attribute_name, cv::Mat& attribute_value) {
	cv::Ptr<hdf::HDF5> h5io = hdf::open(filename);
	if (!h5io->atexists(attribute_name)) {
		std::cout << attribute_name << " is not existed" << std::endl;
	}
	else {
		h5io->atread(attribute_value, attribute_name);
	}
	h5io->close();
}

bool hdf5::checkExist(const std::string& filename, const std::string& parent_name, const std::string& dataset_name) {
	cv::Ptr<hdf::HDF5> h5io = hdf::open(filename);
	bool flag = true;

	if (!h5io->hlexists(parent_name)) {
		flag = false;
		//std::cout << parent_name << " is not existed" << std::endl;
	}
	else if (!h5io->hlexists(parent_name + dataset_name)) {
		flag = false;
		//std::cout << parent_name + dataset_name << " is not existed" << std::endl;
	}
	h5io->close();
	return flag;
}


bool hdf5::insertData(const std::string& filename, const std::string& parent_name, const std::string& dataset_name, const cv::Mat& data) {
	bool flag = true;
	if (!data.empty()) {
		cv::Ptr<hdf::HDF5> h5io = hdf::open(filename);

		if (checkExist(filename, parent_name, dataset_name)) {
			std::string dataset = parent_name + dataset_name;
			std::vector<int> data_size = h5io->dsgetsize(dataset);
			// expand the dataset at row direction
			int offset[2] = { data_size[0],0 };

			if ((h5io->dsgettype(dataset) == data.type()) && (data_size[1] == data.cols)) {
				h5io->dsinsert(data, dataset, offset);
			}

			else {
				flag = false;
				std::cout << std::endl;
				std::cout << " the new data has different size and type with the existed data" << std::endl;
				std::cout << dataset << " insert failed" << std::endl;
			}
		}
		else {
			// first we need to create the parent group
			if (!h5io->hlexists(parent_name)) h5io->grcreate(parent_name);

			std::string dataset = parent_name + dataset_name;
			int chunks[2] = { 1, data.cols };
			// create Unlimited x data.cols, data.type() space, dataset can be expanded unlimted on the row direction
			h5io->dscreate(hdf::HDF5::H5_UNLIMITED, data.cols, data.type(), dataset, hdf::HDF5::H5_NONE, chunks);
			// the first time to write data, offset at row,col direction is 0
			int offset[2] = { 0, 0 };
			h5io->dsinsert(data, dataset, offset);
		}
	}
	return flag;
}