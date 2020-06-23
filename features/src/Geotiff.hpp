#pragma once
#ifndef GEOTIFF_HPP_
#define GEOTIFF_HPP_
#include <iostream>
#include <string>
#include <stdlib.h>
#include "gdal_priv.h"
#include "cpl_conv.h"
#include "gdalwarper.h"
#include <opencv2/opencv.hpp>

/*
This code use GDAL libraries.
Install GDAL in conda:
conda install -c conda-forge gdal
Add conda lib and include path to CmakeLists.txt
*/

using namespace cv;
using namespace std;
typedef std::string String;

class GeoTiff {

public:  
    const char* filename;        // name of Geotiff
    GDALDataset* geotiffDataset;  // Geotiff GDAL datset object. 
    std::array<int, 3> dimensions;            // X,Y, and Z dimensions. 
    int NROWS  , NCOLS , NLEVELS ;     // dimensions of data in Geotiff. 
     
public:

    // define constructor function to instantiate object
    // of this Geotiff class. 
    GeoTiff(const char* tiffname) {
        filename = tiffname;
        GDALAllRegister();
        dimensions = { 0,0,0 };
        // set pointer to Geotiff dataset as class member.  
        geotiffDataset = (GDALDataset*)GDALOpen(filename, GA_ReadOnly);

        // set the dimensions of the Geotiff 
        NROWS = GDALGetRasterYSize(geotiffDataset);
        NCOLS = GDALGetRasterXSize(geotiffDataset);
        NLEVELS = GDALGetRasterCount(geotiffDataset);

    }
    
    // define destructor function to close dataset, 
    // for when object goes out of scope or is removed
    // from memory. 
    ~GeoTiff() 
    {
        // close the Geotiff dataset, free memory for array.
        GDALClose(geotiffDataset);
        GDALDestroyDriverManager();
        
    }
       

    const char* GetFileName() {
        /*
         * function GetFileName()
         * This function returns the filename of the Geotiff.
         */
        return filename;
    }


   
    double GetNoDataValue() {
        /*
         * function GetNoDataValue():
         *  This function returns the NoDataValue for the Geotiff dataset.
         *  Returns the NoData as a double.
         */
        return (double)geotiffDataset->GetRasterBand(1)->GetNoDataValue();
    }

    std::array<int,3> GetDimensions() {
        /*
         * int *GetDimensions():
         *
         *  This function returns a pointer to an array of 3 integers
         *  holding the dimensions of the Geotiff. The array holds the
         *  dimensions in the following order:
         *   (1) number of columns (x size)
         *   (2) number of rows (y size)
         *   (3) number of bands (number of bands, z dimension)
         */
        dimensions[0] = NROWS;
        dimensions[1] = NCOLS;
        dimensions[2] = NLEVELS;
        return dimensions;
    }

// use this functin to get complex HH,HV,VH,VV tiff, z starts from 1
  Mat GetRasterBand(int z) {

        /*
         * function float** GetRasterBand(int z):
         * This function reads a band from a geotiff at a
         * specified vertical level (z value, 1 ...
         * n bands). To this end, the Geotiff's GDAL
         * data type is passed to a switch statement,
         * and the template function GetArray2D (see below)
         * is called with the appropriate C++ data type.
         * The GetArray2D function uses the passed-in C++
         * data type to properly read the band data from
         * the Geotiff, cast the data to float**, and return
         * it to this function. This function returns that
         * float** pointer.
         */
        Mat dst = Mat::zeros(cv::Size(NROWS, NCOLS), CV_8UC1);
        int data_type = GDALGetRasterDataType(geotiffDataset->GetRasterBand(z));
        //cout << "data_type " <<data_type<< endl;
        switch ( data_type) 
        {
        case 1:
           // GDAL GDT_Byte (-128 to 127) - unsigned  char
            dst = Mat::zeros(cv::Size(NROWS, NCOLS), CV_8UC1);
            GetArray2D<unsigned char>(z, dst);
            break;
        case 2:
            // GDAL GDT_UInt16 - short
            dst = Mat::zeros(cv::Size(NROWS, NCOLS), CV_16UC1);
            GetArray2D<unsigned short>(z, dst);
            break;
        case 3:
            // GDT_Int16
             dst = Mat::zeros(cv::Size(NROWS, NCOLS), CV_16SC1);
             GetArray2D<short>(z, dst);
             break;
        case 5:
            // GDT_Int32
            dst = Mat::zeros(cv::Size(NROWS, NCOLS), CV_32SC1);
            GetArray2D<int>(z, dst);
            break;
        case 6:
            // GDT_Float32
            dst = Mat::zeros(cv::Size(NROWS, NCOLS), CV_32FC1);
              GetArray2D<float>(z, dst);
              break;
        case 7:
            // GDT_Float64
            dst = Mat::zeros(cv::Size(NROWS, NCOLS), CV_64FC1);
             GetArray2D<double>(z, dst);
             break;
        // complex value from tiff
        case 8:
            //GDT_CInt16
            dst = Mat_<std::complex<short> >(NROWS, NCOLS);
            GetArray2D<std::complex<short>>(z, dst);
            break;
        case 9:
            //GDT_CInt32
            dst = Mat_<std::complex<int> >(NROWS, NCOLS);
            GetArray2D<std::complex<int>>(z, dst);
            break;
        case 10:
            //GDT_CFloat32
            dst = Mat_<std::complex<float> >(NROWS, NCOLS);
            GetArray2D<std::complex<float>>(z, dst);
            break;
        case 11:
            //GDT_CFloat64
            dst = Mat_<std::complex<double> >(NROWS, NCOLS);
            GetArray2D<std::complex<double>>(z, dst);
            break;
        default: 
            break;
        }
        return dst;
    }

    template<typename T>
    void GetArray2D(int layerIndex, Mat & dst) {

        /*
         * function float** GetArray2D(int layerIndex):
         * This function returns a pointer (to a pointer)
         * for a float array that holds the band (array)
         * data from the geotiff, for a specified layer
         * index layerIndex (1,2,3... for GDAL, for Geotiffs
         * with more than one band or data layer, 3D that is).
         *
         * Note this is a template function that is meant
         * to take in a valid C++ data type (i.e. char,
         * short, int, float), for the Geotiff in question
         * such that the Geotiff band data may be properly
         * read-in as numbers. Then, this function casts
         * the data to a float data type automatically.
         */

         // get the raster data type (ENUM integer 1-12, 
         // see GDAL C/C++ documentation for more details)        
        GDALDataType bandType = GDALGetRasterDataType(
            geotiffDataset->GetRasterBand(layerIndex));

        // get number of bytes per pixel in Geotiff
        int nbytes = GDALGetDataTypeSizeBytes(bandType);

        // allocate pointer to memory block for one row (scanline) 
        // in 2D Geotiff array.  
        T* rowBuff = (T*)CPLMalloc(nbytes * NCOLS);

        for (int row = 0; row < NROWS; row++) {     // iterate through rows

          // read the scanline into the dynamically allocated row-buffer       
            CPLErr e = geotiffDataset->GetRasterBand(layerIndex)->RasterIO(
                GF_Read, 0, row, NCOLS, 1, rowBuff, NCOLS, 1, bandType, 0, 0);
            if (!(e == 0)) {
                cout << "Warning: Unable to read scanline in Geotiff!" << endl;
                exit(1);
            }

            for (int col = 0; col < NCOLS; col++) { // iterate through columns
                dst.at<T>(col,row) = (T)rowBuff[col];
                 
            }
        }
        CPLFree(rowBuff);
        
    }

    Mat GetMat(){
        /*
         * function  Mat GetMat():
         * This function returns a matrix generated from tiff file
         * each channel responses to each band in tiff file if there is no complex values.
         */
        Mat tiff;
        
        if (NLEVELS >1){
            vector<Mat> temp;
            for(int z = 1; z <= NLEVELS; z++) {
                int data_type = GDALGetRasterDataType(geotiffDataset->GetRasterBand(z));
                if (data_type > 7) {
                    cout << "band " << z << " contains complex value" << endl;
                    cout << "please use GetRasterBand(int z) to read mat" << endl;
                    break;
                }
                temp.push_back(GetRasterBand(z));
            }
            merge(temp, tiff);
        }
        else if (NLEVELS ==1) { 
           tiff= GetRasterBand(NLEVELS);
        }
        else {
            cout << "no channel number info" << endl;
            exit(-1);
        }
        return tiff;
    }
};

#endif
