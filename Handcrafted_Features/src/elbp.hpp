#ifndef ELBP_HPP_
#define ELBP_HPP_

#include "opencv2/core/core.hpp"
#include  <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>



// caculate elbp features
// reference: https://github.com/bytefish/opencv/tree/master/lbp

namespace elbp{

    std::string type2str(int type);

    template <typename _Tp>
    void ELBP(const cv::Mat& src, cv::Mat& dst, int radius, int neighbors);

    void ElbpWrapper(const cv::Mat& src, cv::Mat& dst, int radius, int neighbors);

    cv::Mat CaculateElbp(const cv::Mat& src, int radius, int neighbors, bool normed);

}
#endif