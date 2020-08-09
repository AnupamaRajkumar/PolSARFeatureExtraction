#include "cvFeatures.hpp"

using namespace cv;
using namespace std;

/*===================================================================
* Function: GetMP
*
* Summary:
*   Get Morphological profiles (MP) composed of opening-closing by reconstruction
*
* Arguments:
*   const Mat &src - grayscale image
*   const array<int,3> & morph_size - the diameter of circular structureing element,default {1,3,5} 
*
* Returns:
*   3 stacked mp profiles(opening, opening by reconstruction, closing, closing by reconstruction)
=====================================================================
*/
Mat cvFeatures::GetMP(const Mat &src, const array<int,3> & morph_size) {

    Mat dst = src.clone();
    if(src.channels() != 1){
        cvtColor(dst, dst, COLOR_BGR2GRAY);
        normalize(dst, dst, 0, 255, NORM_MINMAX);
    }

    
    std::vector<Mat> temp1 = morph::CaculateMP(dst, morph_size[0]);
    std::vector<Mat> temp2 = morph::CaculateMP(dst, morph_size[1]);
    std::vector<Mat> temp3 = morph::CaculateMP(dst, morph_size[2]);
    cv::Mat result1, result2, result3;
    cv::hconcat(temp1, result1);
    cv::hconcat(temp2, result2);
    cv::hconcat(temp3, result3);
    
    Mat output;
    output.push_back(result1);
    output.push_back(result2);
    output.push_back(result3);

    return output;
}


/*===================================================================
* Function: GetLBP
*
* Summary:
*   Get local binary pattern 
*
* Arguments:
*   const Mat& src -  grayscale
*   int radius  - default 1
*   int neighbors - default 8
*
* Returns:
*   Matrix of Size(src.rows, src.cols) CV_8UC1 
=====================================================================
*/
Mat cvFeatures::GetLBP(const Mat& src, int radius, int neighbors, int histsize  ) {
     
        Mat lbp = elbp::CaculateElbp(src, radius, neighbors, true);

        Mat lbp_hist = GetHistOfMaskArea(lbp, Mat(), 0, 255, histsize, false);
        return lbp_hist.reshape(1,1);
}


/*===================================================================
* Function: GetGLCM
*
* Summary:
*   Calculate energy, contrast, homogenity and entropy of every pixel of the whole img
*
*
* Arguments:
*   const Mat& src -  grayscale
*   vector<Mat>& result - record the CV_8UC1 mat of energy, contrast, homogenity and entropy
*   int winSize - size of Mat Window (only support 5*5, 7*7),default 7
*   GrayLevel level - Destination image's Gray Level (choose in 4/8/16/32),default 16
*
* Returns:
*  Mat  1*32
=====================================================================
*/
Mat cvFeatures::GetGLCM(const Mat& src, int winsize, GrayLevel level, int histsize) {
   
    // to make the returned mat continuous in memory
    Mat output = Mat(1,histsize,CV_8UC1);

    Mat dst = src.clone();
    // src should be nomalized to color images
    normalize(dst, dst, 0, 255, NORM_MINMAX);
    src.convertTo(dst, CV_8UC1);

    // Magnitude Gray Image
    GLCM::GrayMagnitude(dst, dst, level);
    // Calculate Energy, Contrast, Homogenity, Entropy of the whole Image
    Mat Energy_tmp, Contrast_tmp, Homogenity_tmp, Entropy_tmp;
     
    GLCM::CalcuTextureImages(dst, Energy_tmp, Contrast_tmp, Homogenity_tmp, Entropy_tmp, winsize, level, true);
    Mat engergy_hist = GetHistOfMaskArea(Energy_tmp, Mat(), 0, 255, histsize / 4, false);
    Mat Contrast_hist = GetHistOfMaskArea(Contrast_tmp, Mat(), 0, 255, histsize / 4, false);
    Mat Homogenity_hist = GetHistOfMaskArea(Homogenity_tmp, Mat(), 0, 255, histsize / 4, false);
    Mat Entropy_hist = GetHistOfMaskArea(Entropy_tmp, Mat(), 0, 255, histsize/4, false);
    vector<Mat> result;
    result.push_back(engergy_hist);
    result.push_back(Contrast_hist);
    result.push_back(Homogenity_hist);
    result.push_back(Entropy_hist);

    hconcat(result, output);
    return output;
 }



/*===================================================================
* Function: GetMPEG7DCD
*
* Summary:
*   Get MPEG-7 dominant color descriptor (DCD)
*   include three color components and the weight of the most dominant color
*
* Arguments:
*   const Mat& src -  BGR img
*   int numOfColor - number of dominant colors, default 1
*
* Returns:
*    return the weight and color value of each dominant color
=====================================================================
*/
Mat cvFeatures::GetMPEG7DCD(const Mat& src, int numOfColor) {
    Mat dst = src.clone();
    Mat output = Mat(numOfColor,4,CV_8UC1);

    if( src.channels()== 3 && src.type() == CV_8UC3){
        Frame* frame = new Frame(dst.cols, dst.rows, true, true, true);
        frame->setImage(dst);

        // color: DCD, return the weights and color value of each dominant color
        XM::DominantColorDescriptor* dcd = Feature::getDominantColorD(frame, false, false, false);
        // number of dominant colors
        int ndc = dcd->GetDominantColorsNumber();
        if (ndc >= numOfColor) {
            XM::DOMCOL* domcol = dcd->GetDominantColors();
            float weight = 0.0;
            // normalize the weight
            for (int w = 0; w < ndc; w++) {
                weight = weight + domcol[w].m_Percentage;
            }

            for (int w = 0; w < numOfColor; w++) {
                output.at<unsigned char>(w, 0) =static_cast<unsigned char>(domcol[w].m_ColorValue[0]);
                output.at<unsigned char>(w, 1) = static_cast<unsigned char>(domcol[w].m_ColorValue[1]);
                output.at<unsigned char>(w, 2) = static_cast<unsigned char>(domcol[w].m_ColorValue[2]);
                output.at<unsigned char>(w, 3) = static_cast<unsigned char>(domcol[w].m_Percentage / weight);
            }
        }
        // release descriptor
        delete dcd;
    }
    else {
        exit(-1);
        cout << "src should be CV_8UC3" << endl;
    }
    return output.reshape(1,1);
 }

/*===================================================================
* Function: GetMPEG7CSD
*
* Summary:
*   Get MPEG-7 color structure descriptor (CSD)
*
* Arguments:
*   const Mat& src -  BGR img
*   int Size - length of the feature vector,default 32
*
* Returns:

=====================================================================
*/
Mat cvFeatures::GetMPEG7CSD(const Mat& src, int Size) {
    Mat dst = src.clone();
    normalize(src, dst, 0, 255, NORM_MINMAX);
    dst.convertTo(dst, CV_8UC3);
    Frame* frame = new Frame(dst.cols, dst.rows, true, true, true);
    frame->setImage(dst);
    // compute the descriptor
    XM::ColorStructureDescriptor* csd = Feature::getColorStructureD(frame, Size);

    Mat output = Mat(1, Size, CV_8UC1);
    for (unsigned int i = 0; i < csd->GetSize(); ++i) {
        output.at<unsigned char>(0, i) = static_cast<unsigned char>(csd->GetElement(i));
    }
    delete csd;
    return output;
 }

Mat cvFeatures::GetHSV(const Mat& src, int size) {
    Mat dst = src.clone();
    Mat tmp;
    cvtColor(dst, tmp, COLOR_BGR2HSV);
    vector<Mat> channels;
    split(tmp, channels);
    Mat hue = channels.at(0) ;
    Mat saturation = channels.at(1);
    Mat value = channels.at(2) ;
     
    Mat h_hist = GetHistOfMaskArea(hue, Mat(), 0, 179,size/3,true);
    Mat s_hist = GetHistOfMaskArea(saturation, Mat(), 0, 255,size/3,true);
    Mat v_hist = GetHistOfMaskArea(value, Mat(), 0, 255,size/3, true);

    Mat output;
    channels.clear();
    channels.push_back(h_hist.reshape(1,1));
    channels.push_back(s_hist.reshape(1, 1));
    channels.push_back(v_hist.reshape(1, 1));
    hconcat(channels, output);
    return output;
}
/*===================================================================
 * Function: GetStatistic
 *
 * Summary:
 *   Compute median, min, max, mean, std of mask area
 *
 * Arguments:
 *   Mat src -  CV_32FC1
 *
 * Returns:
 *   Mat of Size(1,5)
=====================================================================
*/
Mat cvFeatures::GetStatistic(const Mat& src) {

    Mat output = Mat(1, 5, CV_32FC1);

        vector<float> vec(src.begin<float>(), src.end<float>());
        int size = static_cast<int>(src.total());

        // sort the vector
        std::sort(vec.begin(), vec.end());
        if (size % 2 == 0)
        {
            //median
            output.at<float>(0, 0)= vec[size / 2 - 1] + vec[size / 2] / 2.0f;
        }
        else
        {
            //median
            output.at<float>(0, 0) = vec[size / 2];
        }

        //min
        output.at<float>(0, 1) = vec[0];
        //max
        output.at<float>(0, 2) = vec[size - 1];

        Scalar mean, stddev; //0:1st channel, 1:2nd channel and 2:3rd channel
        meanStdDev(vec, mean, stddev);
        output.at<float>(0, 3) = float(mean[0]);
        output.at<float>(0, 4) = float(stddev[0]);

        return output;
 }
 
Mat cvFeatures::GetHistOfMaskArea(const Mat& src, const Mat& mask, int minVal, int maxVal, int histSize, bool normed)
{
    Mat output;
    // Set the ranges.
    float range[] = { static_cast<float>(minVal), static_cast<float>(maxVal + 1) };
    const float* histRange = { range };
    // calc histogram with mask
    calcHist(&src, 1, 0, mask, output, 1, &histSize, &histRange, true, false);
    // normalize
    if (normed) {
        output /= (int)src.total();
    }
    return output.reshape(1, 1);
}




