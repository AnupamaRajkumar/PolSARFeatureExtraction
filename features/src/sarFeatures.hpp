#pragma once
#ifndef  SARFEATURES_HPP_
#define  SARFEATURES_HPP_
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

namespace polsar {

	//Generate false color image
	// R , G, B
	Mat GetColorImg(const Mat& R, const Mat& G, const Mat& B, bool normed);
	//R: HH+VV, G:HV, B: HH-VV
	Mat GetPauliColorImg(const Mat& hh, const Mat& vv, const Mat& hv);
	Mat GetFalseColorImg(const Mat& hh, const Mat& vv, const Mat& hv, const Mat& vh, bool normed);

	// process complex scattering values
	Mat getComplexAmpl(const Mat& in);
	Mat getComplexAngle(const Mat& in);
	Mat logTransform(const Mat& in);

	// convert CV_32FC2 to Complexd
	Mat_<Complexd> getComplexMat(const Mat& src);
	// get conjugation of complex matrix
	Mat getConj(const Mat& src);
	// get multiplication of two complex matrix
	Mat getMul(const Mat& src1, const Mat& src2);

	void getLexiBasis(const Mat& hh, const Mat& vv, const Mat& hv, vector<Mat>& lexi);
	void getPauliBasis(const Mat& hh, const Mat& vv, const Mat& hv, vector<Mat>& pauli);
	void getCircBasis(const Mat& hh, const Mat& vv, const Mat& hv, vector<Mat>& circ);

	// get up corner elements of coherence or covariance matrix, default winSize 3
	void vec2mat(const vector<Mat>& basis, vector<Mat>& mat, int winSize);

	// get whole matrix from corner elements of C or T matrix
	void getCompleteMatrix(const vector<Mat>& mat, vector<Mat>& complete_mat);

	Mat getPhaseDiff(const Mat& hh, const Mat& vv);

	// get up corner elements 0f coherence matrix T from Pauli basis
	void GetCoherencyT(const vector<Mat>& pauli, vector<Mat>& coherence, int winSize = 3);

	// get up corner elements 0f covariance matrix C from lexicographic basis
	void GetCovarianceC(const vector<Mat>& lexi, vector<Mat>& covariance, int winSize = 3);

	// Compute the span image from the trace of the covariance or coherence matrix for the pixel
	void createSpanImage(const Mat& m00, const Mat& m11, const Mat& m22, Mat& span);

	// coherent decomposition 
	// get krogager decomposition (sphere,diplane,helix) from circular basis
	void GetKrogagerDecomp(const vector<Mat>& circ, vector<Mat>& decomposition);

	// coherent decomposition 
	// get pauli decompostition (|alpha|,|beta|,|gamma|) from Pauli basis
	void GetPauliDecomp(const vector<Mat>& pauli, vector<Mat>& decomposition);

	// model based decomposition
	// get Ps, Pd, Pv from covariance matrix
	void GetFreemanDurdenDecomp(const vector<Mat>& covariance, vector<Mat>& decomposition);
	void freemanDurdenDecomp(const Mat_<Complexd>& covariance, vector<double>& result);

	// model based decomposition
	// get Ps, Pd, Pv, Pc from covariance matrix
	void GetYamaguchi4Decomp(const vector<Mat>& coherence, const vector<Mat>& covariance, vector<Mat>& decomposition);
	void yamaguchi4Decomp(const Mat_<Complexd>& coherence, const Mat_<Complexd>& covariance, vector<double>& result);

	// eigenvector based decomposition
	// get H, a, A from  coherence matrix
	void GetCloudePottierDecomp(const vector<Mat>& coherence, vector<Mat>& decomposition);
	void cloudePottierDecomp(Mat_<Complexd>& coherence, vector<double>& result);

	// dichotomy of the kennaugh matrix
	// get A0, B0, B, C, D, E, F, G, H from coherence matrix
	void GetHuynenDecomp(const vector<Mat> & coherence, vector<Mat> & decomposition);
	void huynenDecomp(const Mat_<Complexd>& coherence, vector<double>& result);
	
	// vector<Mat>& data is hh,vv,hv or vv, vh or hh,hv
	// Rect&cell is the patch area
	// get statistical (min,max,mean,median,std) on polsar parameters
	void getStatisticFeature(const vector<Mat>& data, vector<Mat>& result);
	void GetFullPolStat(Mat& hh, const Mat& vv, const Mat& hv, vector<Mat>& result);

	// get upper triangle matrix elements of C, T
	void GetCTelements(const Mat& hh, const Mat& vv, const Mat& hv, vector<Mat>& result);

	//restore 3*3 covariance or coherence mat from upcorner vector elements
	Mat restoreMatrix(const vector<Mat>& upcorner, int row, int col);
}

#endif