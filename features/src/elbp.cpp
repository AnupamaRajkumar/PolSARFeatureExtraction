# include "elbp.hpp"

using namespace std;
using namespace cv;

std::string elbp::type2str(int type) {
    std::string r;
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);
    switch (depth) {
    case CV_8U:  r = "CV_8U"; break;
    case CV_8S:  r = "CV_8S"; break;
    case CV_16U: r = "CV_16U"; break;
    case CV_16S: r = "CV_16S"; break;
    case CV_32S: r = "CV_32S"; break;
    case CV_32F: r = "CV_32F"; break;
    case CV_64F: r = "CV_64F"; break;
    default:     r = "User"; break;
    }
    r += "C";
    r += (chans + '0');
    return r;
}

template <typename _Tp>
void elbp::ELBP(const Mat& src, Mat& dst, int radius, int neighbors)
{

    for (int n = 0; n < neighbors; n++) {
        // sample points
        float x = static_cast<float>(-radius * sin(2.0 * CV_PI * n / static_cast<float>(neighbors)));
        float y = static_cast<float>(radius * cos(2.0 * CV_PI * n / static_cast<float>(neighbors)));
        // relative indices
        int fx = static_cast<int>(floor(x));
        int fy = static_cast<int>(floor(y));
        int cx = static_cast<int>(ceil(x));
        int cy = static_cast<int>(ceil(y));
        // fractional part
        float ty = y - fy;
        float tx = x - fx;
        // set interpolation weights
        float w1 = (1 - tx) * (1 - ty);
        float w2 = tx * (1 - ty);
        float w3 = (1 - tx) * ty;
        float w4 = tx * ty;
        // iterate through your data
        for (int i = radius; i < src.rows - radius; i++) {
            for (int j = radius; j < src.cols - radius; j++) {
                // calculate interpolated value
                float t = static_cast<float>(w1 * src.at<_Tp>(i + fy, j + fx) + w2 * src.at<_Tp>(i + fy, j + cx) + w3 * src.at<_Tp>(i + cy, j + fx) + w4 * src.at<_Tp>(i + cy, j + cx));
                // floating point precision, so check some machine-dependent epsilon
                dst.at<_Tp>(i - radius, j - radius) += ((t > src.at<_Tp>(i, j)) || (std::abs(t - src.at<_Tp>(i, j)) < std::numeric_limits<float>::epsilon())) << n;
            }
        }
    }
}

void elbp::ElbpWrapper(const Mat& src, Mat& dst, int radius, int neighbors)
{
    std::string type = type2str(src.type());
    if (type == "CV_8SC1") {
        dst = Mat(src.rows, src.cols, CV_8SC1);
        ELBP<char>(src, dst, radius, neighbors);
    }
    else if (type == "CV_8UC1") {
        dst = Mat(src.rows, src.cols, CV_8UC1);
        ELBP<unsigned char>(src, dst, radius, neighbors);
    }
    else if (type == "CV_16SC1") {
        dst = Mat(src.rows, src.cols, CV_16SC1);
        ELBP<short>(src, dst, radius, neighbors);
    }
    else if (type == "CV_16UC1") {
        dst = Mat(src.rows, src.cols, CV_16UC1);
        ELBP<unsigned short>(src, dst, radius, neighbors);
    }
    else if (type == "CV_32SC1") {
        dst = Mat(src.rows, src.cols, CV_32SC1);
        ELBP<int>(src, dst, radius, neighbors);
    }
    else if (type == "CV_32FC1") {
        dst = Mat(src.rows, src.cols, CV_32FC1);
        ELBP<float>(src, dst, radius, neighbors);
    }
    else {
        dst = Mat(src.rows, src.cols, CV_64FC1);
        ELBP<double>(src, dst, radius, neighbors);
    }

}

/*===================================================================
* Function: CaculateElbp
 *
 * Summary:
 *   Calculate elbp from the whole image.
 *
 * Arguments:
 *   Mat src - source Matrix (Window Mat)
 *   int radius - The radius of the circle,which allows us to account for different scales
 *   int neighbors - The number of points in a circularly symmetric neighborhood to consider
 *   bool normed - whether to normalized to 0-255 CV_8UC1
 *
 * Returns:
 *  Matrix of Size(src.size()), single channel
=====================================================================
*/
Mat elbp::CaculateElbp(const Mat& src, int radius, int neighbors, bool normed) {
    Mat dst;
    if (src.channels() > 1)
    {
        Mat tmp;
        cvtColor(src, tmp, COLOR_BGR2GRAY);
        ElbpWrapper(tmp, dst, radius, neighbors);
    }
    else {
        ElbpWrapper(src, dst, radius, neighbors);
    }

    if (normed) {
        cv::normalize(dst, dst, 0, 255, NORM_MINMAX, CV_8UC1);
    }
    return dst;
}