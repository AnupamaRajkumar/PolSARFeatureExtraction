#include "mp.hpp"
#include <opencv2/opencv.hpp>
using namespace cv;

auto* mp::matToArray(const Mat& image) {
    if (image.channels() != 1) {
        cvtColor(image, image, COLOR_BGR2GRAY);
    }
    // flatten the Mat.
    uint totalElements = static_cast<int>(image.total()) * image.channels();
    Mat flat = image.reshape(1, totalElements);
    if (!image.isContinuous()) {
        flat = flat.clone();
    }
    auto* ptr = flat.data;
    return ptr;
}

// compute opening-closing by reconstruction from image
// example:https://de.Mathworks.com/help/images/marker-controlled-watershed-segmentation.html
Mat mp::CaculateMP(const Mat& src, int morph_size) {
    //convert img to grayscale
    Mat dst;
    if (src.channels() != 1) {
        cvtColor(src, dst, COLOR_BGR2GRAY);
    }
    else {
        src.copyTo(dst);
    }
    equalizeHist(dst, dst);
    //imshow("image", dst);
    // waitKey(0);

    Mat element = getStructuringElement(MORPH_ELLIPSE, cv::Size(2 * morph_size + 1, 2 * morph_size + 1), cv::Point(morph_size, morph_size));


    //erode and reconstruct ( opening-by-reconstruction )
    Mat Iobr = Mat(Size(dst.size()), dst.type());
    erode(dst, Iobr, element);
    auto* ptr = matToArray(Iobr);
    mp::Reconstruct(ptr, matToArray(dst), dst.cols, dst.rows);
    //restore cv Mat
    Mat dst2 = Mat(dst.rows, dst.cols, dst.type(), ptr);
   // imshow("openning by reconstruction: ", dst2);
   // waitKey(0);

     //dilate and reconstruct (closing-by-Reconstruction)
    Mat Icbr = Mat(Size(dst2.size()), dst2.type());
    dilate(dst2, Icbr, element);
    // imcomplement
    dst2 = 255 - dst2;
    Icbr = 255 - Icbr;
    auto* ptr2 = matToArray(Icbr);
    mp::Reconstruct(ptr2, matToArray(dst2), dst2.cols, dst2.rows);
    //restore cv Mat
    Mat dst3 = Mat(dst.rows, dst.cols, dst.type(), ptr2);
    // imcomplement
    dst3 = 255 - dst3;
   // imshow("opening-closing by reconstruction: ", dst3);
    //waitKey(0);
   return dst3;
}