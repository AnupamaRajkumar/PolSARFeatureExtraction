#include "specklefilter.hpp"
#include "sarFeatures.hpp"
#include<vector>
#include<iostream>
#include "Utils.h"


using namespace cv;
using namespace std;

void RefinedLee::filterFullPol(Mat& hh,  Mat& vv,  Mat& hv) {
    vector<Mat> lexi;
    polsar::getLexiBasis(hh, vv, hv, lexi);
    vector<Mat> upcorner_covariance;
    polsar::GetCovarianceC(lexi, upcorner_covariance);
    Mat span = Mat(Size(hh.size()), CV_32FC1);
    Mat m00 = upcorner_covariance[0];
    Mat m11 = upcorner_covariance[3];
    Mat m22 = upcorner_covariance[5];

    polsar::createSpanImage(m00,m11,m22, span);
    filterFullPol(hh, vv, hv, span);
}

//override
void RefinedLee::filterFullPol(Mat& hh, Mat& vv, Mat& hv, Mat&span) {

    Mat neighborPixelValues = Mat(filterSize, filterSize, CV_32FC1);
    Mat neighborSpanValues = Mat(filterSize, filterSize, CV_32FC1);

    vector<Mat> temp(6);
    extractChannel(hh, temp[0], 0);
    extractChannel(hh, temp[1], 1);
    extractChannel(vv, temp[2], 0);
    extractChannel(vv, temp[3], 1);
    extractChannel(hv, temp[4], 0);
    extractChannel(hv, temp[5], 1);
    for (int i = 0; i < 6; i++) {
        Mat t = temp.at(i);
        Mat temp_output = Mat(Size(hh.size()), CV_32FC1);
        for (int x = 0; x < hh.rows; x++) {
            for (int y = 0; y < hh.cols; y++) {
                int n = getLocalData(x, y, t, span, neighborPixelValues, neighborSpanValues);
                if (n < filterSize * filterSize) {
                    temp_output.at<float>(x, y) = computePixelValueUsingLocalStatistics(neighborPixelValues, n);
                }
                else {
                    temp_output.at<float>(x, y) = computePixelValueUsingEdgeDetection(neighborPixelValues, neighborSpanValues);
                }
            }
        }
        temp.at(i) = temp_output;
    }
    vector<Mat> output(2);
    output.at(0) = temp.at(0);
    output.at(1) = temp.at(1);
    cv::merge(output, hh);

    output.at(0) = temp.at(2);
    output.at(1) = temp.at(3);
    cv::merge(output, vv);

    output.at(0) = temp.at(4);
    output.at(1) = temp.at(5);
    cv::merge(output, hv);

}




/**
     * Compute filtered pixel value using Local Statistics filter.
     *
     * @param neighborPixelValues The pixel values in the neighborhood.
     * @return The filtered pixel value.
     */
float RefinedLee::computePixelValueUsingLocalStatistics(const Mat& neighborPixelValues, int numSamples) {

    int halfFilterSize = neighborPixelValues.rows / 2;
    // here y is the pixel amplitude or intensity and x is the pixel reflectance before degradation
    float meanY = getLocalMeanValue(neighborPixelValues, numSamples);
    float varY = getLocalVarianceValue(neighborPixelValues, numSamples,meanY);
    if (varY == 0.0) {
        return 0.0;
    }

    float varX = (varY - meanY * meanY * sigmaVSqr) / (1 + sigmaVSqr);
    if (varX < 0.0) {
        varX = 0.0;
    }
    float b = varX / varY;
    return meanY + b * (neighborPixelValues.at<float>(halfFilterSize,halfFilterSize) - meanY);
}

/**
 * Compute filtered pixel value using refined Lee filter.
 *
 * @param neighborPixelValues The pixel values in the neighborhood.
 * @param neighborSpanValues  The span image pixel values in the neighborhood.
 * @return The filtered pixel value.
 */
float RefinedLee::computePixelValueUsingEdgeDetection(const Mat & neighborPixelValues, const Mat& neighborSpanValues) {
    
    int halfFilterSize = filterSize / 2;

    Mat subAreaMeans = Mat(3, 3, CV_32FC1);
    computeSubAreaMeans(neighborSpanValues, subAreaMeans);

    int d = getDirection(subAreaMeans);

    Mat spanPixels;
    int numPixels = getNonEdgeAreaPixelValues(neighborSpanValues, d, spanPixels);

    float meanY = getLocalMeanValue(spanPixels, numPixels);
    float varY = getLocalVarianceValue(spanPixels, numPixels,meanY);
    if (varY == 0.0) {
        return 0.0;
    }

    float varX = (varY - meanY * meanY * sigmaVSqr) / (1 + sigmaVSqr);
    if (varX < 0.0) {
        varX = 0.0;
    }
    float b = varX / varY;

    Mat elemPixels;
    numPixels= getNonEdgeAreaPixelValues(neighborPixelValues, d, elemPixels);
    float meanZ = getLocalMeanValue(elemPixels, numPixels);

    return meanZ + b * (neighborPixelValues.at<float>(halfFilterSize,halfFilterSize) - meanZ);
}


/**
* Get pixel intensities in a filter size rectangular region centered at the given pixel.
*
* @param x                  X coordinate of the given pixel.
* @param y                  Y coordinate of the given pixel.
* @param filterSize         Sliding window size.
* @param src                The source img (single channel)
* @param span                The span img (single channel)
* @param neighborPixelValues 2 - D array holding the pixel values.
* @param neighborSpanValues 2 - D array holding the pixel values.
* @return The number of valid pixels.
**/
int RefinedLee::getLocalData( int x,  int y,const Mat & src,const Mat & span, Mat & neighborPixelValues, Mat & neighborSpanValues) {

    int halfSize = filterSize / 2;
    int numSamples = 0;
    int maxX = src.rows;
    int maxY = src.cols;

    for (int j = 0; j < filterSize; ++j) {
        int yj = y - halfSize + j;
        if (yj < 0 || yj >= maxY) {
            for (int i = 0; i < filterSize; ++i) {
                neighborPixelValues.at<float>(i, j) = NonValidPixelValue;
                neighborSpanValues.at<float>(i, j) = NonValidPixelValue;
            }
            continue;
        }
        for (int i = 0; i < filterSize; ++i) {
            int xi = x - halfSize + i;
            if (xi < 0 || xi >= maxY) {
                neighborPixelValues.at<float>(i, j) = NonValidPixelValue;
                neighborSpanValues.at<float>(i, j) = NonValidPixelValue;
                continue;
            }
            else {
                neighborPixelValues.at<float>(i, j) = src.at<float>(xi, yj);
                neighborSpanValues.at<float>(i, j) = span.at<float>(xi, yj);
                numSamples++;
            }
        }
    }
    return numSamples;
}

/**
     * Get the mean value of pixel intensities in a given rectangular region.
     *
     * @param neighborValues The pixel values in the given rectangular region.
     * @param numSamples     The number of samples.
     * @return mean The mean value.
**/
float RefinedLee::getLocalMeanValue(const Mat & neighborValues, int numSamples) {

        float mean = 0.0;
        for (int i = 0; i < neighborValues.rows; i++){
            for (int j = 0; j < neighborValues.cols; j++){
                float v = neighborValues.at<float>(i, j);
                if (v != NonValidPixelValue) {
                    mean += v;
                }
            }
        }
        mean /= numSamples;

        return mean;
    }

    /**
     * Get the variance of pixel intensities in a given rectangular region.
     *
     * @param neighborValues The pixel values in the given rectangular region.
     * @param numSamples     The number of samples.
     * @param mean           the mean of neighborValues.
     * @param noDataValue    Place holder for no data value.
     * @return var The variance value.
     * @throws OperatorException If an error occurs in computation of the variance.
     */
 float RefinedLee::getLocalVarianceValue(const Mat& neighborValues, int numSamples, float mean) {
            
     float var = 0.0;
     if (numSamples > 1) {

         for (int i = 0; i < neighborValues.rows; i++) {
             for (int j = 0; j < neighborValues.cols; j++) {
                 float v = neighborValues.at<float>(i, j);
                 if (v != NonValidPixelValue) {
                     float diff = v - mean;
                     var += diff * diff;
                 }
             }
             var /= (numSamples - 1.0);
         }
     }

        return var;
    }





/**
     * Compute mean values for the 3x3 sub-areas in the sliding window.
     *
     * @param stride              Stride for shifting sub-window within the sliding window.
     * @param subWindowSize       Size of sub-area.
     * @param neighborPixelValues The pixel values in the sliding window.
     * @param subAreaMeans        The 9 mean values.
     */
 void RefinedLee::computeSubAreaMeans( const Mat& neighborPixelValues, Mat& subAreaMeans) {
    
    int subWindowSizeSqr = subWindowSize * subWindowSize;
    for (int j = 0; j < 3; j++) {
        int y0 = j * stride;
        for (int i = 0; i < 3; i++) {
            int x0 = i * stride;

            float mean = 0.0;
            for (int y = y0; y < y0 + subWindowSize; y++) {
                for (int x = x0; x < x0 + subWindowSize; x++) {
                    mean += neighborPixelValues.at<float>(y,x);
                }
            }
            subAreaMeans.at<float>(j,i) = mean / subWindowSizeSqr;
        }
    }
}


/**
     * Get gradient direction.
     *
     * @param subAreaMeans The mean values for the 3x3 sub-areas in the sliding window.
     * @return The direction.
     */
 int RefinedLee::getDirection(const Mat & subAreaMeans) {

    std::array<float,4> gradient ;
    // horizontal gradient
    gradient[0] = subAreaMeans.at<float>(0,2) + subAreaMeans.at<float>(1,2) + subAreaMeans.at<float>(2,2) -
        subAreaMeans.at<float>(0,0) - subAreaMeans.at<float>(1,0) - subAreaMeans.at<float>(2,0);

    gradient[1] = subAreaMeans.at<float>(0,1) + subAreaMeans.at <float>(0,2) + subAreaMeans.at<float>(1,2) -
        subAreaMeans.at<float>(1,0)- subAreaMeans.at <float>(2,0) - subAreaMeans.at<float>(2,1);

    gradient[2] = subAreaMeans.at<float>(0,0) + subAreaMeans.at<float>(0,1) + subAreaMeans.at<float>(0,2) -
        subAreaMeans.at<float>(2,0)- subAreaMeans.at<float>(2,1) - subAreaMeans.at<float>(2,2);

    gradient[3] = subAreaMeans.at<float>(0,0) + subAreaMeans.at<float>(0,1) + subAreaMeans.at < float>(1,0) -
        subAreaMeans.at<float>(1,2) - subAreaMeans.at<float>(2,1) - subAreaMeans.at<float>(2,2);

    int direction = 0;
    float maxGradient = -1.0;
    for (int i = 0; i < 4; i++) {
        float absGrad = std::abs(gradient[i]);
        if (maxGradient < absGrad) {
            maxGradient = absGrad;
            direction = i;
        }
    }

    // if the maximum gradient is small than zero, switch to the opposite direction
    if (gradient[direction] > 0.0) {
        direction += 4;
    }

    return direction;
}


 /**
     * Get pixel values from the non-edge area indicated by the given direction.
     *
     * @param neighborPixelValues The pixel values in the filterSize by filterSize neighborhood.
     * @param d                   The direction index.
     * @param *pixels              The Mat of pixels.
     */
 int RefinedLee::getNonEdgeAreaPixelValues(const Mat& neighborPixelValues, int d, Mat &pixels) {

     int filterSize = neighborPixelValues.rows;
     int halfFilterSize = filterSize / 2;
     pixels = Mat::zeros(filterSize, filterSize, CV_32FC1);
     int k = 0;

     switch (d) {
     case 0: {

         int k = 0;
         for (int y = 0; y < filterSize; y++) {
             for (int x = halfFilterSize; x < filterSize; x++) {
                 pixels.at<float>(y, x)= neighborPixelValues.at<float>(y, x);
                 k++;
             }
         }
         break;
     }
     case 1: {

         for (int y = 0; y < filterSize; y++) {
             for (int x = y; x < filterSize; x++) {
                 pixels.at<float>(y, x) = neighborPixelValues.at<float>(y, x);
                 k++;
             }
         }
         break;
     }
     case 2: {

         for (int y = 0; y <= halfFilterSize; y++) {
             for (int x = 0; x < filterSize; x++) {
                 pixels.at<float>(y, x) = neighborPixelValues.at<float>(y, x);
                 k++;
             }
         }
         break;
     }
     case 3: {

         for (int y = 0; y < filterSize; y++) {
             for (int x = 0; x < filterSize - y; x++) {
                 pixels.at<float>(y, x) = neighborPixelValues.at<float>(y, x);
                 k++;
             }
         }
         break;
     }
     case 4: {

         for (int y = 0; y < filterSize; y++) {
             for (int x = 0; x <= halfFilterSize; x++) {
                 pixels.at<float>(y, x) = neighborPixelValues.at<float>(y, x);
                 k++;
             }
         }
         break;
     }
     case 5: {

         for (int y = 0; y < filterSize; y++) {
             for (int x = 0; x < y + 1; x++) {
                 pixels.at<float>(y, x) = neighborPixelValues.at<float>(y, x);
                 k++;
             }
         }
         break;
     }
     case 6: {

         for (int y = halfFilterSize; y < filterSize; y++) {
             for (int x = 0; x < filterSize; x++) {
                 pixels.at<float>(y, x) = neighborPixelValues.at<float>(y, x);
                 k++;
             }
         }
         break;
     }
     case 7: {

         for (int y = 0; y < filterSize; y++) {
             for (int x = filterSize - 1 - y; x < filterSize; x++) {
                 pixels.at<float>(y, x) = neighborPixelValues.at<float>(y, x);
                 k++;
             }
         }
         break;
     }
     }
     return k;
 }
