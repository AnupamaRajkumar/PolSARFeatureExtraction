#ifndef ELBP_HPP_
#define ELBP_HPP_

#include "opencv2/core/core.hpp"
#include  <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

// caculate elbp features
namespace elbp{

    std::string type2str(int type);

    template <typename _Tp>
    void ELBP(const Mat& src, Mat& dst, int radius, int neighbors);

    void ElbpWrapper(const Mat& src, Mat& dst, int radius, int neighbors);

    Mat CaculateElbp(const Mat& src, int radius, int neighbors, bool normed);

}
#endif