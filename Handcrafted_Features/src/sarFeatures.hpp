#pragma once
#ifndef  SARFEATURES_HPP_
#define  SARFEATURES_HPP_
#include <opencv2/opencv.hpp>
#include "cvFeatures.hpp"
#include <complex>
#include <algorithm>   



namespace polsar {

	cv::Mat GetColorImg(const cv::Mat& R, const cv::Mat& G, const cv::Mat& B, bool normed);
	//R: HH+VV, G:HV, B: HH-VV
	cv::Mat GetPauliColorImg(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv);
	cv::Mat GetFalseColorImg(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv, const cv::Mat& vh, bool normed);

	// process complex scattering values
	cv::Mat getComplexAmpl(const cv::Mat& in);
	cv::Mat getComplexAngle(const cv::Mat& in);
	cv::Mat logTransform(const cv::Mat& in);

	// convert CV_32FC2 to Complexf
	cv::Mat_<cv::Complexf> getComplexMat(const cv::Mat& src);
	// get conjugation of complex matrix
	cv::Mat getConj(const cv::Mat& src);
	// get multiplication of two complex matrix
	cv::Mat getMul(const cv::Mat& src1, const cv::Mat& src2);

	void getLexiBasis(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv, std::vector<cv::Mat>& lexi);
	void getPauliBasis(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv, std::vector<cv::Mat>& pauli);
	void getCircBasis(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv, std::vector<cv::Mat>& circ);

	// get up corner elements of coherence or covariance matrix, default winSize 3
	void vec2mat(const std::vector<cv::Mat>& basis, std::vector<cv::Mat>& mat, int winSize);

	// get whole matrix from corner elements of C or T matrix
	void getCompleteMatrix(const std::vector<cv::Mat>& mat, std::vector<cv::Mat>& complete_mat);


	// get up corner elements 0f coherence matrix T from Pauli basis
	void GetCoherencyT(const std::vector<cv::Mat>& pauli, std::vector<cv::Mat>& coherence, int winSize = 3);

	// get up corner elements 0f covariance matrix C from lexicographic basis
	void GetCovarianceC(const std::vector<cv::Mat>& lexi, std::vector<cv::Mat>& covariance, int winSize = 3);

	// Compute the span image from the trace of the covariance or coherence matrix for the pixel
	void createSpanImage(const cv::Mat& m00, const cv::Mat& m11, const cv::Mat& m22, cv::Mat& span);

	// coherent decomposition 
	// get krogager decomposition (sphere,diplane,helix) from circular basis
	void GetKrogagerDecomp(const std::vector<cv::Mat>& circ, std::vector<cv::Mat>& decomposition);

	// coherent decomposition 
	// get pauli decompostition (|alpha|,|beta|,|gamma|) from Pauli basis
	void GetPauliDecomp(const std::vector<cv::Mat>& pauli, std::vector<cv::Mat>& decomposition);

	// model based decomposition
	// get Ps, Pd, Pv from covariance matrix
	void GetFreemanDurdenDecomp(const std::vector<cv::Mat>& covariance, std::vector<cv::Mat>& decomposition);
	void freemanDurdenDecomp(const cv::Mat_<cv::Complexf>& covariance, std::vector<float>& result);

	// model based decomposition
	// get Ps, Pd, Pv, Pc from covariance matrix
	void GetYamaguchi4Decomp(const std::vector<cv::Mat>& coherence, const std::vector<cv::Mat>& covariance, std::vector<cv::Mat>& decomposition);
	void yamaguchi4Decomp(const cv::Mat_<cv::Complexf>& coherence, const cv::Mat_<cv::Complexf>& covariance, std::vector<float>& result);

	// eigenvector based decomposition
	// get H, Anisotropy, Alpha, eigenValue from coherence matrix
	void GetCloudePottierDecomp(const std::vector<cv::Mat>& coherence, std::vector<cv::Mat>& decomposition);
	void cloudePottierDecomp(cv::Mat_<cv::Complexf>& coherence, std::vector<float>& result);
	void eigenDecomposition(int n, const cv::Mat& HMr, const cv::Mat& HMi, cv::Mat& EigenVectRe, cv::Mat& EigenVectIm, std::vector<float>& EigenVal);
	 
	// get upper triangle matrix elements of C, T
	void GetCTelements(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat& hv, std::vector<cv::Mat>& result);

	//restore 3*3 covariance or coherence mat from upcorner vector elements
	cv::Mat restoreMatrix(const std::vector<cv::Mat>& upcorner, int row, int col);

	cv::Mat getCoPolarizationRatio(const cv::Mat& hh, const cv::Mat& vv, int winSize);
	cv::Mat getDePolarizationRatio(const cv::Mat& hh, const cv::Mat& vv, const cv::Mat & hv, int winSize);

	// get <band1*conju(band2)>
	cv::Mat calcuCoherenceOfPol(const cv::Mat& band1, const cv::Mat& band2, int winSize);
	cv::Mat getPhaseDiff(const cv::Mat& vv, const cv::Mat& vh);
}

#endif