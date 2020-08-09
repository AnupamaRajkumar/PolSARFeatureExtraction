/*=================================================================
 * Calculate GLCM(Gray-level Co-occurrence cv::Matrix) By OpenCV.
 *
 * Copyright (C) 2017 Chandler Geng. All rights reserved.
 *
 *     This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as published
 * by the Free Software Foundation; either version 2 of the License, or (at
 * your option) any later version.
 *
 *     This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 *     You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc., 59
 * Temple Place, Suite 330, Boston, MA 02111-1307 USA
===================================================================
*/

#ifndef GLCM_HPP_
#define GLCM_HPP_

#include <iostream>
#include <opencv2/opencv.hpp>
#include  <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Math.h>



// Gray Level (Choose in 4/8/16)
enum class GrayLevel
{
    GRAY_4 = 4,
    GRAY_8 = 8,
    GRAY_16= 16,
    GRAY_32 = 32
};


// Gray Value Statistical Direction
// (Choose in 0°, 45°, 90°, 135°)
enum class GrayDirection
{
    DIR_0 = 0,
    DIR_45 = 45 ,
    DIR_90 = 90,
    DIR_135 =135
};


// Point out R, G, B Channel of a Image
enum class RGBChannel
{
    CHANNEL_R = 2,
    CHANNEL_G = 1,
    CHANNEL_B = 0
};

// struct including Texture Eigenvalues
struct TextureEValues
{
    
    float energy;
   
    float contrast;
  
    float homogenity;

    float entropy;
};

namespace GLCM
{
   
    // Extract a channel from RGB Image
    void getOneChannel(cv::Mat src, cv::Mat& dstChannel, RGBChannel channel = RGBChannel::CHANNEL_R);

   
    // Magnitude all pixels of Gray Image, and Magnitude Level can be chosen in 4/8/16;
    void GrayMagnitude(cv::Mat src, cv::Mat& dst, GrayLevel level = GrayLevel::GRAY_8);

   
    // Calculate the GLCM of one cv::Mat Window according to one Statistical Direction.
    void CalcuOneGLCM(cv::Mat src, cv::Mat &dst, int src_i, int src_j, int size, GrayLevel level = GrayLevel::GRAY_8, GrayDirection direct = GrayDirection::DIR_0);

    
    //   Normalize the Martix, make all pixels of cv::Mat divided by the sum of all pixels of  Mat, then get Probability  Matrix.
    void NormalizeMat(cv::Mat src, cv::Mat& dst);

   
    // Calculate Texture Eigenvalues of One Window cv::Mat, which is including Energy, Contrast, Homogenity, Entropy.
    void CalcuEValue(cv::Mat src, TextureEValues& EValue, bool ToCheckMat = false);

    //Calculate the averaged Texture Eigenvalues of the whole image, which is including Energy, Contrast, Homogenity, Entropy
    void CalcuTextureEValue(cv::Mat src, TextureEValues& EValue, int size, GrayLevel level);

    //Caculate Texture Eigenvalues of the whole image
    void CalcuTextureImages(cv::Mat src, cv::Mat& imgEnergy, cv::Mat& imgContrast, cv::Mat& imgHomogenity, cv::Mat& imgEntropy,
                            int size = 5, GrayLevel level = GrayLevel::GRAY_8, bool ToAdjustImg = false);
}


#endif