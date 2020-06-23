#include "sen12ms.hpp"

// record the class values
std::array<unsigned char, 17>  IGBP_label = { 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 };
// LCCS_LC, LCCS_LU,LCCS_SH merged into LCCS
std::array<unsigned char, 26>  LCCS_label = { 11,12,13,14,15,16,10,21,22,30,31,32,40,41,43,42,27,50,36,9,25,35,51,2,1,3 };
/*===================================================================
 * Function: getFalseColorImage
 *
 * Summary:
 *   Generate false color image from SAR data;
 *
 * Arguments:
 *   Mat src - 2 channel matrix(values in dB) from tiff file
 *   bool normed - normalized to 0-255 
 *
 * Returns:
 *   3 channel matrix: R: VV, G:VH, B: VV/VH
=====================================================================
*/
Mat sen12ms::getFalseColorImage(const Mat& src, bool normed) {
    vector<Mat>  Channels;
    split(src, Channels);

    Mat R = abs(Channels[0]); //VV
    Mat G = abs(Channels[1]);  //VH
    
    

    Mat B = Mat::zeros(src.rows, src.cols, CV_32FC1); //VV/VH
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (Channels[1].at<float>(i, j) != 0) {
                B.at<float>(i, j) =  Channels[0].at<float>(i, j)  /  Channels[1].at<float>(i, j) ;
            }
        }
    }
    Mat dst = Mat::zeros(src.rows, src.cols, CV_32FC3);
    vector<Mat> temp;
    temp.push_back(B);
    temp.push_back(G);
    temp.push_back(R);
    merge(temp, dst);
    if (normed) {
        normalize(dst, dst, 0, 255, NORM_MINMAX);
        dst.convertTo(dst, CV_8UC3);
    }
    return dst;
}


/*===================================================================
 * Function: getLabelMap
 *
 * Summary:
 *   Merge LCCS_LC, LCCS_LU,LCCS_SH into LCCS class
 *   Generate IGBP or LCCS label map from the ground truth 
 *
 * Arguments:
 *   Mat src - 4 channel matrix from groundtruth file
 *   Mat& labelMap
 *
 * Returns:
 *   void
=====================================================================
*/
void sen12ms::getLabelMap(const Mat& lc, Mat& labelMap) {

    labelMap = Mat(Size(lc.size()), CV_8UC1);
    vector<Mat> temp(lc.channels());
    split(lc, temp);
    if( mask_type == MaskType::IGBP){
        labelMap = temp[0];
    } else if (mask_type == MaskType::LCCS){
        Mat LCCS_LC = temp[1];
        Mat LCCS_LU = temp[2];
        Mat LCCS_SH = temp[3];

        for (int i = 0; i < lc.rows; i++) {
            for (int j = 0; j < lc.cols; j++) {
                if (LCCS_LC.at<unsigned char>(i, j) != 0) {
                    labelMap.at<unsigned char>(i, j) = LCCS_LC.at<unsigned char>(i, j);
                }
                else if (LCCS_LU.at<unsigned char>(i, j) != 0) {
                    labelMap.at<unsigned char>(i, j) = LCCS_LU.at<unsigned char>(i, j);
                }
                else {
                    labelMap.at<unsigned char>(i, j) = LCCS_SH.at<unsigned char>(i, j);
                }
            }
        }
    }
}
 


/*===================================================================
 * Function: getMask
 *
 * Summary:
 *   Create Masks for each class category
 *
 * Arguments:
 *   Mat & labelMap  
 *   vector<Mat> & list_masks - Destination Mask Mat 
 *   vector<unsigned char> list_classValue
 *
 * Returns:
 *  void
=====================================================================
*/
void sen12ms::getMask(const Mat& labelMap, vector<Mat>& list_masks,  vector<unsigned char> &list_classValue) {
    //get IGBP mask
    if(mask_type ==MaskType::IGBP){
        for (int i = 0; i < IGBP_label.size(); i++) {
            vector<std::pair<int, int> > ind;
            if (findLandClass(labelMap, ind, IGBP_label[i])) {
                list_classValue.push_back(IGBP_label[i]);
                Mat tmp = Mat::zeros(labelMap.rows, labelMap.cols, CV_8UC1);
                for (auto const& p : ind) {
                    tmp.at<unsigned char>(p.first, p.second) = IGBP_label[i];
                }
                list_masks.push_back(tmp);
            }
        }
    }
    // get LCCS_mask
    if (mask_type == MaskType::IGBP) {
        for (int i = 0; i < LCCS_label.size(); i++) {
            vector<std::pair<int, int> > ind;
            if (findLandClass(labelMap, ind, LCCS_label[i])) {
                list_classValue.push_back(LCCS_label[i]);
                Mat tmp = Mat::zeros(labelMap.rows, labelMap.cols, CV_8UC1);
                for (auto const& p : ind) {
                    tmp.at<unsigned char>(p.first, p.second) = LCCS_label[i];
                }
                list_masks.push_back(tmp);
            }
        }
    }
}

/*===================================================================
 * Function: findLandClass
 *
 * Summary:
 *   check if cetain class type existed in a class category
 *
 * Arguments:
 *   Mat & labelMap  
 *   vector<std::pair<int, int> > &ind - record the index of the class type
 *   const int &landclass: value of the class type
 *
 * Returns:
 *  bool
=====================================================================
*/
 bool sen12ms::findLandClass(const Mat& labelMap, vector<std::pair<int, int> > &ind, const unsigned char&landclass) {
     bool flag = false;
    for (int i = 0; i < labelMap.rows; i++) {
        for (int j = 0; j < labelMap.cols; j++) {
            if (labelMap.at<unsigned char>(i,j) == landclass) {
                ind.push_back(std::make_pair(i, j));
                flag = true;
            }
        }
    }
    return flag;
}

 

/*===================================================================
 * Function: GeneratePNG
 *
 * Summary:
 *   convert tiff files to png format for images and masks
 *   the list of RGB images are in list_images.txt
 *   the list of IGBP masks are in list_IGBP_masks.txt
 *   the list of LCCS masks are in list_LCCS_masks.txt
 *
 * Arguments:
 *   string outputpath - the folder path to store all the pngs
 *   MaskType mask_type - IGBP or LCCS
 * Returns:
 *  void
=====================================================================
*/
 void sen12ms::GeneratePNG(const string &outputpath) {
     
     fstream list_images;
     string list_images_path = outputpath + "\\" + "list_images.txt";
     list_images.open(list_images_path, ios::out | ios::trunc);

     fstream listofmasks;

     if(mask_type ==MaskType::IGBP){
     string list_IGBP_path = outputpath + "\\" + "list_IGBP_masks.txt";
     listofmasks.open(list_IGBP_path, ios::out | ios::trunc);
     } else if(mask_type == MaskType::LCCS) {
         string list_LCCS_path = outputpath + "\\" + "list_LCCS_masks.txt";
         listofmasks.open(list_LCCS_path, ios::out | ios::trunc);
     }

     if (list_images.is_open()) {
         for (const auto &tp: s1FileList) {
                 if (tp.empty()) { cout << "empty line find" << endl; break; }
                 //get the filename from path without extension
                 string base_filename = tp.substr(tp.find_last_of("/\\") + 1);
                 size_t position = base_filename.find(".");
                 string fileName = (string::npos == position) ? base_filename : base_filename.substr(0, position);

                 Mat patch =  Utils::readTiff(tp);
                 Mat colorImg =  getFalseColorImage(patch, true);
                 string outputpng = outputpath + "\\" + fileName + ".png";
                 imwrite(outputpng, colorImg);
                 list_images << outputpng << endl;
         }
         list_images.close();
     }


     if (listofmasks.is_open()) {
         for (const auto &tp: lcFileList) {
                 if (tp.empty()) { cout << "empty line find" << endl; break; }
                 //get the filename from path without extension
                 string base_filename = tp.substr(tp.find_last_of("/\\") + 1);
                 size_t position = base_filename.find(".");
                 string fileName = (string::npos == position) ? base_filename : base_filename.substr(0, position);
                  
                 //Replace String In Place
                 size_t pos = 0;
                 string search = "_lc_", replace = "_s1_";
                 while ((pos = fileName.find(search, pos)) != std::string::npos) {
                     fileName.replace(pos, search.length(), replace);
                     pos += replace.length();
                 }

                 Mat lc_mat =  Utils::readTiff(fileName);

                 vector<Mat> list_mask;
                 vector<unsigned char> list_classValue; //store the class value
                 Mat labelMap = Mat(Size(lc_mat.size()), CV_8UC1);
                 getLabelMap(lc_mat, labelMap);
                 getMask(labelMap, list_mask, list_classValue);
                 string outputpng;
                 string outputFolder;
                 if (mask_type == MaskType::IGBP) {
                     outputFolder = outputpath + "\\" + fileName + "_IGBP";
                 }
                 else if(mask_type == MaskType::LCCS){
                     outputFolder = outputpath + "\\" + fileName + "_LCCS";
                 }
                  
                 int status = _mkdir(outputFolder.c_str());
                 if (status < 0) {
                     cout << "failed to create mask folder for p" << fileName << endl;
                     break;
                 }
                 else {
                     for (int j = 0; j < list_classValue.size(); j++) {
                         outputpng = outputFolder + "\\" + to_string(list_classValue[j]) + ".png";
                         imwrite(outputpng, list_mask[j]);
                         listofmasks << outputpng << endl;
                     }
                 }
         }
         listofmasks.close();
     }
     cout << "Generate PNG files done." << endl;
 }

 


 /*===================================================================
 * Function: LoadBatchToMemeory
 *
 * Summary:
 *   load tiff files to vector<Mat> list_images, vector<Mat> labelMap
 *
 * Arguments:
 *   int batch   - which batch to load, eg:batch =0 means load the first batch
 *   
 * Returns:
 *  void
=====================================================================
*/
 void sen12ms::LoadBatchToMemeory(int batch) {
     if (batchSize > 0) {
         int total_batch = int(s1FileList.size() / batchSize);
         if (batch > total_batch) {
             cout << "batch must be smaller than " << total_batch << endl;
             exit(-1);
         }

         int start = batch * batchSize;
         int end = (batch + 1) * batchSize;
         if (end > static_cast<int>(s1FileList.size())) {
             end = s1FileList.size();
         }

         for (int i = start; i < end; i++) {
             Mat s1_mat = Utils::readTiff(s1FileList[i]);
             Mat unnomarlizedImg = getFalseColorImage(s1_mat, false);
             list_images->at(i) = unnomarlizedImg;
             Mat lc_mat = Utils::readTiff(lcFileList[i]);
             Mat labelMap = Mat(Size(lc_mat.size()), CV_8UC1);
             getLabelMap(lc_mat, labelMap);
             list_labelMaps->at(i) = labelMap;
         }
 
         cout << "Load " << batchSize << " images and its masks to memory" << endl;
     }
     else { 
         cout << "Please set batch size bigger than zero." << endl;
     }
 }
  

 void sen12ms::LoadAllToMemory() {
     for (int i = 0; i < s1FileList.size(); i++) {
         Mat s1_mat = Utils::readTiff(s1FileList[i]);
         Mat unnomarlizedImg = getFalseColorImage(s1_mat, true);
         list_images->at(i) = unnomarlizedImg;
         Mat lc_mat = Utils::readTiff(lcFileList[i]);
         Mat labelMap = Mat(Size(lc_mat.size()), CV_8UC1);
         getLabelMap(lc_mat, labelMap);
         list_labelMaps->at(i) = labelMap;
     }
     cout << "Load " << s1FileList.size() << " images and its masks to memory" << endl;
 }

 /*===================================================================
 * Function: GetPatches
 *
 * Summary:
 *   get samples for each mask area
 *
 * Arguments:
 *   vector<Mat>& patches - store patches drawed from each image
 *   vector<unsigned char>& classValue  - store the class type for each patch
 *
 * Returns:
 *  void
=====================================================================
*/
 void sen12ms::GetPatches(vector<Mat>& patches, vector<unsigned char>& classValue) {
     if (list_images->empty()) { cout << "Please load data to memory first! " << endl;  exit(-1); }
     for (int i = 0; i < list_images->size(); i++) {
         Mat img = list_images->at(i);
         Mat label_map = list_labelMaps->at(i);

         vector<Mat> masks;
         vector<unsigned char> mask_labels;
         getMask(label_map, masks, mask_labels);
         // generate patches from mask area
         for (int j = 0; j < masks.size(); j++) {
             getSamples(img, masks[j], mask_labels[j], patches, classValue);
         }
     }
 }
 
  
 /*===================================================================
 * Function: loadFileList
 *
 * Summary:
 *   Load tiff file list
 *
 * Arguments:
 *   string s1FileList - the txt file of all s1 files path
 *   string lcFileList - the txt file list of all lc files path
 *
 * Returns:
 *  void
=====================================================================
*/
 void sen12ms::loadFileList(const string& s1FileListPath, const string& lcFileListPath) {
     fstream s1File;
     s1File.open(s1FileListPath, ios::in);

     std::ifstream lcFile(lcFileListPath);
     const std::string lcFileString((std::istreambuf_iterator<char>(lcFile)), std::istreambuf_iterator<char>());


     if (s1File.is_open()) {
         string s1FilePath;
         while (getline(s1File, s1FilePath)) {
             size_t pos = 0;
             if (s1FilePath.empty()) { cout << "empty line find" << endl; break; }
             // if the path is not tiff file
             if (s1FilePath.find("tif", pos) == std::string::npos) { continue; }
              
             // to make sure each s1File is corresponding to lcFile
             // get the lc filename, only need to replace all the _s1_ to _lc_
             string lcFileName = s1FilePath.substr(s1FilePath.find_last_of("/\\") + 1);
             //Replace replace all the _s1_ to _lc
             string search = "_s1_", replace = "_lc_";
             while ((pos = lcFileName.find(search, pos)) != std::string::npos) {
                 lcFileName.replace(pos, search.length(), replace);
                 pos += replace.length();
             }

             // get the absolute file path of this lc file from lcFileString
             string tmp = lcFileString.substr(0, lcFileString.find(lcFileName));
             string lcFilePath = tmp.substr(tmp.find_last_of("\n") + 1) + lcFileName;
             if (!lcFilePath.empty()) {
                 s1FileList.push_back(s1FilePath);
                 lcFileList.push_back(lcFilePath);
             }
         }
         s1File.close();
     }
     cout << "list size of s1 files: " << s1FileList.size() << endl;
     cout << "list size of lc files: " << lcFileList.size() << endl;
 }


 /*===================================================================
 * Function: GetClassName
 *
 * Summary:
 *  get the class name from class value
 *
 * Arguments:
 *   int classValue 
 *   MaskType mask_type: choose in IGBP or LCCS
 *
 * Returns:
 *  string - class name
=====================================================================
*/
 string sen12ms::GetClassName(signed char classValue){
     string class_name;
     std::map<unsigned char, string> IGBP = {
       {0,"Unclassified"},
       {1,"Evergreen Needleleaf Forests"},
{2,"Evergreen Broadleaf Forests"},
{3,"Deciduous Needleleaf Forests"},
{4,"Deciduous Broadleaf Forests"},
{5,"Mixed Forests"},
{6,"Closed (Dense) Shrublands"},
{7,"Open (Sparse) Shrublands"},
{8,"Woody Savannas"},
{9,"Savannas"},
{10,"Grasslands "},
{11,"Permanent Wetlands"},
{12,"Croplands"},
{13,"Urban and Built-Up Lands"},
{14,"Cropland/Natural Vegetation Mosaics"},
{15,"Permanent Snow and Ice"},
{16,"Barren"},
{17,"Water Bodies"}
     };

     std::map<unsigned char, string> LCCS = {
       {0,"Unclassified"},
{1,"Barren"},
{2,"Permanent Snow and Ice"},
{3,"Water Bodies"},
{9,"Urban and Built-Up Lands"},
{10,"Dense Forests"},
{11,"Evergreen Needleleaf Forests"},
{12,"Evergreen Broadleaf Forests"},
{13,"Deciduous Needleleaf Forests"},
{14,"Deciduous Broadleaf Forests"},
{15,"Mixed Broadleaf/Needleleaf Forests"},
{16,"Mixed Broadleaf Evergreen/Deciduous Forests"},
{21,"Open Forests "},
{22,"Sparse Forests"},
{25,"Forest/Cropland Mosaics"},
{27,"Woody Wetlands"},
{30,"Natural Herbaceous"},
{30,"Grasslands"},
{31,"Dense Herbaceous"},
{32,"Sparse Herbaceous"},
{35,"Natural Herbaceous/Croplands Mosaics"},
{36,"Herbaceous Croplands"},
{40,"Shrublands"},
{41,"Closed (Dense) Shrublands"},
{42,"Shrubland/Grassland Mosaics"},
{43,"Open (Sparse) Shrublands"},
{50,"Herbaceous Wetlands"},
{51,"Tundra"}
     };
 
     if (mask_type == MaskType::IGBP) {
         class_name = IGBP[classValue];
     }
     else {
         class_name = LCCS[classValue];
     }
     return class_name;
 }

  



 // input: img and its label_map, output: samples with its label
 void sen12ms::getSamples(const Mat& img, const Mat& mask, const unsigned char& mask_label, vector<Mat>& samples, vector<unsigned char>& sample_labels)
 {
     int cnt = 0;
     // get the masks and labels for the image
     vector<Point> samplePoints;
     Utils::getSafeSamplePoints(mask, samplePointNum, sampleSize, samplePoints);
     if(!samplePoints.empty()){
         //draw patches centered at each sample point
         for (const auto& p : samplePoints)
         {
             int start_x = int(p.x) - sampleSize / 2;
             int start_y = int(p.y) - sampleSize / 2;
             Rect roi = Rect(start_x, start_y, sampleSize, sampleSize);
             Mat tmp = img(roi).clone();
             samples.push_back(tmp);
             sample_labels.push_back(mask_label);
             cnt++;
         }
         cout << cnt <<" samples drawed for class " << int(mask_label) << ": " << GetClassName(mask_label) << endl;
     }
 }

 

 
  