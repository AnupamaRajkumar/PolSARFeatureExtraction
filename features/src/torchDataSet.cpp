#include <ATen/ATen.h>
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <tuple>
#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;
using namespace std;

class torchDataset : public torch::data::datasets::Dataset<torchDataset> {
private:
    // Declare 2 vectors of tensors for images and labels
    std::vector<torch::Tensor> images, labels;
public:

    // Constructor
    torchDataset(vector<Mat>& list_images, vector<unsigned char>& classValues) {
        // process_data will write to images, labels
        process_data(list_images, classValues);
    };

    // Override get() function to return tensor at location index
    torch::data::Example<> get(size_t index) override {
        torch::Tensor sample_img = images.at(index);
        torch::Tensor sample_label = labels.at(index);
        return { sample_img.clone(), sample_label.clone() };
    };

    // Return the length of data
    torch::optional<size_t> size() const override {
        return labels.size();
    };

private:
    /* Convert and Load image to tensor from location argument */
    torch::Tensor read_img_data(Mat &img) {
         
       // Return tensor form of the image
       // cv::resize(img, img, cv::Size(1920, 1080), cv::INTER_CUBIC);
       // std::cout << "Sizes: " << img.size() << std::endl;
        torch::Tensor img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte);
        img_tensor = img_tensor.permute({ 2, 0, 1 }); // Channels x Height x Width

        return img_tensor.clone();
    }

    /* Converts label to tensor type in the integer argument */
    torch::Tensor read_label(unsigned char label) {
        // Read label here
        // Convert to tensor and return
        int test = int(label);
        torch::Tensor label_tensor = torch::full({ 1 },test, torch::kInt);
        return label_tensor.clone();
    }

    /* Loads images to tensor type in the string argument */
    void  process_data(vector<Mat> &list_images, vector<unsigned char> &classValues) {
        
        for ( auto& c : list_images) {
            images.push_back(read_img_data(c));
        }
        for (auto& v : classValues) {
            labels.push_back(read_label(v));
        }
    }
 
};