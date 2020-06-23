#include "sarFeatures.hpp"
#include "cvFeatures.hpp"
#include <complex>
#include <Eigen/Eigenvalues>
#include <algorithm>    
 


using namespace cv;
using namespace std;

static constexpr double m_Epsilon = 1e-6f;
const double PI_F = 3.14159265358979f;
const double CONST_180_PI = 180.0 / PI_F;

Mat polsar::getComplexAmpl(const Mat& in) {

	Mat out;
	vector<Mat> channels;

	split(in, channels);
	pow(channels[0], 2, channels[0]);
	pow(channels[1], 2, channels[1]);
    out = channels[0] + channels[1];
	pow(out, 0.5, out);
	
	return out;
}

Mat polsar::logTransform(const Mat& in) {

	Mat out = in.clone();
	if (in.channels() == 2) {
		out = getComplexAmpl(in);
	}
	out = out + 1;
	log(out, out);

	return out;
}

Mat polsar::getComplexAngle(const Mat& in) {

	vector<Mat> channels;
	split(in, channels);
	Mat amp, out;
	cartToPolar(channels[0], channels[1], amp, out);

	return out;

}

void polsar::getLexiBasis(const Mat& hh, const Mat& vv, const Mat& hv, vector<Mat>& lexi) {
	lexi.push_back(hh);
	lexi.push_back(sqrt(2.0) * hv);
	lexi.push_back(vv);
}


void polsar::getPauliBasis(const Mat& hh, const Mat& vv, const Mat& hv, vector<Mat>& pauli) {
	pauli.push_back( (hh + vv) / sqrt(2.0));
	pauli.push_back( (hh - vv) / sqrt(2.0));
	pauli.push_back( hv * sqrt(2.0));
	//cout << "hh(5,5) value + vv(5,5) value: " << hh.at<Vec2d>(5, 5)[0] + vv.at<Vec2d>(5, 5)[0] << endl;
	//cout <<"pauli(5,5)* sqrt(2) value: "<< pauli[0].at<Vec2d>(5, 5)[0]*sqrt(2.0) << endl;
}

/**
* Create Span image.
*
* @param sourceBands         the input bands
* @param sourceTileRectangle The source tile rectangle.
* @param span                The span image.
*/
// The pixel value of the span image is given by the trace of the covariance or coherence matrix for the pixel.
void polsar::createSpanImage(const Mat& m00, const Mat& m11, const Mat& m22, Mat& span) {

	span = Mat(Size(m00.size()), m00.type());

	span = (m00 + m11 + m22) / 4.0;
}

void polsar::getCircBasis(const Mat& hh, const Mat& vv, const Mat& hv, vector<Mat>& circ) {

	// i*s_hv = i*(a+ib) = ai-b = -b+ia
	Mat a, b;
	cv::extractChannel(hv, a, 0);
	cv::extractChannel(hv, b, 1);
	vector<Mat> channels;
	channels.push_back(-1.0 * b);
	channels.push_back(a);
	Mat i_hv;
	merge(channels, i_hv);

	// s_rr = (s_hh - s_vv + i*2*s_hv)/2
	Mat srr = (hh - vv + 2.0 * i_hv) / 2.0;

	// s_ll = (s_vv - s_hh + i*2*s_hv)/2
	Mat sll = (vv - hh + 2.0 * i_hv) / 2.0;

	// s_lr = i*(s_hh + s_vv)/2
	Mat temp = hh + vv;
	cv::extractChannel(temp, a, 0);
	cv::extractChannel(temp, b, 1);
	channels.clear();
	channels.push_back(-1.0 * b);
	channels.push_back(a);
	Mat slr;
	merge(channels, slr);
	slr = 0.5 * slr;

	circ.push_back(sll);
	circ.push_back(srr);
	circ.push_back(slr);

}


void polsar::vec2mat(const vector<Mat>& basis, vector<Mat>& mat, int winSize) {
	Mat m00, m01, m02, m11, m12, m22;

	mulSpectrums(basis.at(0), basis.at(0), m00, 0, true); //|k_0 | ^ 2
	mulSpectrums(basis.at(0), basis.at(1), m01, 0, true); //k_0*conj(k_1)
	mulSpectrums(basis.at(0), basis.at(2), m02, 0, true); //k_0*conj(k_2)
	mulSpectrums(basis.at(1), basis.at(1), m11, 0, true); //k_1|^2
	mulSpectrums(basis.at(1), basis.at(2), m12, 0, true); //k_1*conj(k_2)
	mulSpectrums(basis.at(2), basis.at(2), m22, 0, true); //|k_2|^2 

	//m00 = getMul(basis.at(0), getConj(basis.at(0))); // k_0*conj(k_1)
	//m01 = getMul(basis.at(0), getConj(basis.at(1))); // k_0*conj(k_1)
	//m02 = getMul(basis.at(0), getConj(basis.at(2))); // k_0*conj(k_2)
	//m11 = getMul(basis.at(1), getConj(basis.at(1))); // k_1*conj(k_1)
	//m12 = getMul(basis.at(1), getConj(basis.at(2))); // k_1*conj(k_2)
	//m22 = getMul(basis.at(2), getConj(basis.at(2))); // k_2*conj(k_2)


	cv::blur(m00, m00, Size(winSize, winSize));
	cv::blur(m01, m01, Size(winSize, winSize));
	cv::blur(m02, m02, Size(winSize, winSize));
	cv::blur(m11, m11, Size(winSize, winSize));
	cv::blur(m12, m12, Size(winSize, winSize));
	cv::blur(m22, m22, Size(winSize, winSize));

	// the real part is the squared amplitude
	Mat m00_dat, m11_dat, m22_dat;
	extractChannel(m00, m00_dat, 0);
	extractChannel(m11, m11_dat, 0);
	extractChannel(m22, m22_dat, 0);

	mat.push_back(m00_dat);
	mat.push_back(m01);
	mat.push_back(m02);
	mat.push_back(m11_dat);
	mat.push_back(m12);
	mat.push_back(m22_dat);
}

// get the whole C or T matrix from up corner elements
void polsar::getCompleteMatrix(const vector<Mat>& mat, vector<Mat>& complete_mat) {
	Mat m00 = mat[0];
	Mat m01 = mat[1];
	Mat m02 = mat[2];
	Mat m11 = mat[3];
	Mat m12 = mat[4];
	Mat m22 = mat[5];
	Mat m10 = getConj(m01);
	Mat m20 = getConj(m02);
	Mat m21 = getConj(m12);
	complete_mat.push_back(m00);
	complete_mat.push_back(m01);
	complete_mat.push_back(m02);
	complete_mat.push_back(m10);
	complete_mat.push_back(m11);
	complete_mat.push_back(m12);
	complete_mat.push_back(m20);
	complete_mat.push_back(m21);
	complete_mat.push_back(m22);
}

Mat polsar::GetColorImg(const Mat& R, const Mat& G, const Mat& B, bool normed) {
	vector<Mat> Channels;
	Mat output;
	Channels.push_back(B);
	Channels.push_back(G);
	Channels.push_back(R);
	merge(Channels, output);
	if (normed) {
		normalize(output, output, 0, 255, NORM_MINMAX);
		output.convertTo(output, CV_8UC3);
	}
	return output;
}

Mat polsar::GetFalseColorImg(const Mat& hh, const Mat& vv, const Mat& hv, const Mat& vh, bool normed)
{
	Mat R, G, B;
	if (!hh.empty() && !vv.empty() && !hv.empty()) {
		R = logTransform(getComplexAmpl(hh));
		G = logTransform(getComplexAmpl(hv));
		B = logTransform(getComplexAmpl(vv));
	}
	else if (!vv.empty() && !vh.empty()) {
		R = logTransform(getComplexAmpl(vv));
		G = logTransform(getComplexAmpl(vh));
		B = Mat::zeros(vv.rows, vv.cols, R.type()); //VV/VH
		B = R / G;
	}
	//else if (!hh.empty() && !hv.empty()) {
	//	R = logTransform(getComplexAmpl(hh));
	//	G = logTransform(getComplexAmpl(hv));
	//	B = Mat::zeros(hh.rows, hh.cols, R.type()); //HH/HV
	//	B = R / G;
	//}
	else {
		cout << "input pol data is empty" << endl;
		return Mat();
	}
	return GetColorImg(R, G, B,normed);
}



//R: |HH+VV|, G:|HV|, B: |HH-VV|
Mat polsar::GetPauliColorImg(const Mat& hh, const Mat& vv, const Mat& hv) {

	Mat R = logTransform(getComplexAmpl(hh + vv));
	Mat G = logTransform(getComplexAmpl(hv));
	Mat B = logTransform(getComplexAmpl(hh - vv));
	return GetColorImg(R, G, B, true);
}

// get the complex conjugation of a 2 channel matrix
Mat polsar::getConj(const Mat& src) {
	Mat temp = Mat(Size(src.size()), CV_64FC2);
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++) {
			temp.at<Vec2d>(i, j)[0] = src.at<Vec2d>(i, j)[0];
			temp.at<Vec2d>(i, j)[1] = src.at<Vec2d>(i, j)[1] * -1.0;
		}
	return temp;
}

// get the complex muliplication of two 2 channel matrix
Mat polsar::getMul(const Mat& src1, const Mat& src2) {
	Mat temp = Mat(Size(src1.size()), CV_64FC2);
	for (int i = 0; i < src1.rows; i++)
		for (int j = 0; j < src1.cols; j++) {
			// (a+bi)(c+di) = ac-bd + (ad + bc)i
			double a = src1.at<Vec2d>(i, j)[0];
			double b = src1.at<Vec2d>(i, j)[1];
			double c = src2.at<Vec2d>(i, j)[0];
			double d = src2.at<Vec2d>(i, j)[1];
			temp.at<Vec2d>(i, j)[0] = a * c - b * d;
			temp.at<Vec2d>(i, j)[1] = a * d + b * c;
		}
	return temp;
}

// get the phase diff of two CV_64FC2 matrix
Mat polsar::getPhaseDiff(const Mat& hh, const Mat& vv) {
	Mat temp = getMul(hh, getConj(vv));
	return getComplexAngle(temp);
}

// convert CV_64FC2 to Complexd
Mat_<Complexd> polsar::getComplexMat(const Mat& src) {
	Mat dst = Mat_<Complexd>(Size(src.size()));
	if (src.channels() == 2) {
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				dst.at<Complexd>(i, j).re = src.at<Vec2d>(i, j)[0];
				dst.at<Complexd>(i, j).im = src.at<Vec2d>(i, j)[1];
			}
		}
	}
	else if (src.channels() == 1) {
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				dst.at<Complexd>(i, j).re = src.at<double>(i, j);
				dst.at<Complexd>(i, j).im = 0.0f;
			}
		}
	}
	return dst;
}

void polsar::GetKrogagerDecomp(const vector<Mat>& circ, vector<Mat>& decomposition) {
	// sphere, diplane, helix-decomposition
	Mat k_s, k_d, k_h;
	Mat s_ll_amp, s_rr_amp, s_lr_amp;
	s_ll_amp = getComplexAmpl(circ.at(0));
	s_rr_amp = getComplexAmpl(circ.at(1));
	s_lr_amp = getComplexAmpl(circ.at(2));

	// k_s = |s_lr|
	k_s = logTransform( s_lr_amp);

	// k_d = min( |s_ll|, |s_rr| )
	min(s_ll_amp, s_rr_amp, k_d);
	k_d = logTransform(k_d);

	// k_h = | |s_ll| - |s_rr| |
	k_h = abs(s_ll_amp - s_rr_amp);
	k_h = logTransform(k_h);

	decomposition.push_back(k_s);
	decomposition.push_back(k_d);
	decomposition.push_back(k_h);
}

// reference
// https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb/-/blob/develop/Modules/Filtering/Polarimetry/include/otbReciprocalPauliDecompImageFilter.h
void polsar::GetPauliDecomp(const vector<Mat>& pauli, vector<Mat> & decomposition) {

	decomposition.push_back(logTransform(getComplexAmpl(pauli.at(0))));
	decomposition.push_back(logTransform(getComplexAmpl(pauli.at(1))));
	decomposition.push_back(logTransform(getComplexAmpl(pauli.at(2))));
}

// reference:
// https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb/-/blob/develop/Modules/Filtering/Polarimetry/include/otbReciprocalHuynenDecompImageFilter.h
void polsar::huynenDecomp(const Mat_<Complexd>& covariance, vector<double>& result) {
	double A0, B0, B, C, D, E, F, G, H;
	A0 = covariance.at<Complexd>( 0, 0).re / 2.0;
	B0 = (covariance.at<Complexd>(1, 1)+covariance.at<Complexd>(2, 2)).re/ 2.0;
	B = covariance.at<Complexd>(1, 1).re - B0;
	C = covariance.at<Complexd>(0, 1).re;
	D = -1.0*covariance.at<Complexd>(0, 1).im;
	E = covariance.at<Complexd>(1, 2).re;
	F = covariance.at<Complexd>(1, 2).im;
	G = covariance.at<Complexd>(0, 2).im;
	H = covariance.at<Complexd>(0, 2).re;
	
	result.push_back(A0);
	result.push_back(B0);
	result.push_back(B);
	result.push_back(C);
	result.push_back(D);
	result.push_back(E);
	result.push_back(F);
	result.push_back(G);
	result.push_back(H);
}


void polsar::GetHuynenDecomp(const vector<Mat>& upcorner_coherence, vector<Mat>& decomposition) {
	Mat A0, B0, B, C, D, E, F, G, H;
	extractChannel(upcorner_coherence[0], A0, 0);
	A0 = A0 / 2.0;
    
	extractChannel(upcorner_coherence[3]+ upcorner_coherence[5], B0, 0);
	B0 =  B0 / 2.0;

	extractChannel(upcorner_coherence[3], B, 0);
	B = B - B0;

	extractChannel(upcorner_coherence[1], C, 0);

	extractChannel(upcorner_coherence[1], D, 1);
	D = -1.0 * D;

	extractChannel(upcorner_coherence[4], E, 0);

	extractChannel(upcorner_coherence[4], F, 1);

	extractChannel(upcorner_coherence[2], G, 1);

	extractChannel(upcorner_coherence[2], H, 0);

	decomposition.push_back(A0);
	decomposition.push_back(B0);
	decomposition.push_back(B);
	decomposition.push_back(C);
	decomposition.push_back(D);
	decomposition.push_back(E);
	decomposition.push_back(F);
	decomposition.push_back(G);
	decomposition.push_back(H);
}

// reference
// https://github.com/senbox-org/s1tbx/blob/master/rstb/rstb-op-polarimetric-tools/src/main/java/org/csa/rstb/polarimetric/gpf/decompositions/FreemanDurden.java
void polsar::freemanDurdenDecomp(const Mat_<Complexd>& covariance, vector<double>& result) {

	double fd, fv, fs, c11, c13Re, c13Im, c33, alphaRe, alphaIm, betaRe, betaIm;
	// compute fv from m22 and subtract fv from m11, m13, m33
	fv = 4.0f * covariance.at<Complexd>(1, 1).re;
	c11 = covariance.at<Complexd>(0, 0).re - fv * 3.0 / 8.0;
	c13Re = covariance.at<Complexd>(0, 2).re - fv / 8.0;
	c13Im = covariance.at<Complexd>(0, 2).im;
	c33 = covariance.at<Complexd>(2, 2).re - fv * 3.0 / 8.0;
	double a1 = c11 * c33;

	if (c11 <= m_Epsilon || c33 <= m_Epsilon) {
		fs = 0.0f;
		fd = 0.0f;
		alphaRe = 0.0f;
		alphaIm = 0.0f;
		betaRe = 0.0f;
		betaIm = 0.0f;
	}
	else {

		double a2 = c13Re * c13Re + c13Im * c13Im;
		if (a1 < a2) {
			double c13 = std::sqrt(a2);
			c13Re = std::sqrt(a1) * c13Re / c13;
			c13Im = std::sqrt(a1) * c13Im / c13;
		}
		// get sign of Re(C13), if it is minus, set beta = 1; else set alpha = -1
		if (c13Re < 0.0) {

			betaRe = 1.0;
			betaIm = 0.0;
			fs = std::abs((a1 - c13Re * c13Re - c13Im * c13Im) / (c11 + c33 - 2 * c13Re));
			fd = std::abs(c33 - fs);
			alphaRe = (c13Re - fs) / fd;
			alphaIm = c13Im / fd;

		}
		else {

			alphaRe = -1.0;
			alphaIm = 0.0;
			fd = std::abs((a1 - c13Re * c13Re - c13Im * c13Im) / (c11 + c33 + 2 * c13Re));
			fs = std::abs(c33 - fd);
			betaRe = (c13Re + fd) / fs;
			betaIm = c13Im / fs;
		}
	}

	// compute Ps, Pd and Pv
	double ps = fs * (1 + betaRe * betaRe + betaIm * betaIm);
	double pd = fd * (1 + alphaRe * alphaRe + alphaIm * alphaIm);
	double pv = fv;
	result.push_back(ps);
	result.push_back(pd);
	result.push_back(pv);
}


// reference:
// https://gitlab.orfeo-toolbox.org/orfeotoolbox/otb/-/blob/develop/Modules/Filtering/Polarimetry/include/otbReciprocalHAlphaImageFilter.h
void polsar::cloudePottierDecomp(Mat_<Complexd>& coherence, vector<double>& result) {

	Eigen::Map<Eigen::Matrix<std::complex<double>, 3, 3, Eigen::RowMajor>> eigen_mat(coherence.ptr<std::complex<double>>(), coherence.rows, coherence.cols);
	Eigen::ComplexEigenSolver<Eigen::MatrixXcd> ces;
	ces.compute(eigen_mat);

	vector<int> V;// indicating positions
	vector<double> realEigenValues;
	// extract the first component of each eigen vector sorted by eigen value decrease order
	vector<complex<double>> cos_alpha;

	int N = ces.eigenvalues().rows();
	for (int i = 0; i < N; i++) { V.push_back(i); }
	// sort eigen values in decreasing order, and record the original index in V
	sort(V.begin(), V.end(), [&](int i, int j) {return ces.eigenvalues()[i].real() > ces.eigenvalues()[j].real(); });

	for(auto & i: V){
		realEigenValues.push_back(ces.eigenvalues()[i].real());
		cos_alpha.push_back(ces.eigenvectors()(0, i));
	}
	//check the size of eigen values
	if (N == 2) {
		realEigenValues.push_back(0.0);
		cos_alpha.push_back(complex<double>(0.0, 0.0));
	}
	if (N == 1) {
		realEigenValues.push_back(0.0);
		realEigenValues.push_back(0.0);
		cos_alpha.push_back(complex<double>(0.0, 0.0));
		cos_alpha.push_back(complex<double>(0.0, 0.0));
	}

	// Entropy estimation
	double totalEigenValues = 0.0;
	double p[3];
	double plog[3];
	double entropy;
	double alpha;
	double anisotropy;
	for (unsigned int k = 0; k < 3; ++k)
	{
		realEigenValues[k] = std::max(realEigenValues[k], 0.0);
		totalEigenValues += realEigenValues[k];
	}


	for (unsigned int k = 0; k < 3; ++k)
	{
		p[k] = realEigenValues[k] / totalEigenValues;

		if (p[k] < m_Epsilon) // n=log(n)-->0 when n-->0
			plog[k] = 0.0;
		else
			plog[k] = -p[k] * log(p[k]) / log(3.0);
	}

	entropy = 0.0;
	for (unsigned int k = 0; k < 3; ++k)
		entropy += plog[k];

	// Anisotropy estimation
	anisotropy = (realEigenValues[1] - realEigenValues[2]) / (realEigenValues[1] + realEigenValues[2] + m_Epsilon);

	// alpha estimation
	double val0, val1, val2;
	double a0, a1, a2;

	val0 = std::abs(cos_alpha[0]);
	a0 = acos(std::abs(val0)) * CONST_180_PI;

	val1 = std::abs(cos_alpha[1]);
	a1 = acos(std::abs(val1)) * CONST_180_PI;

	val2 = std::abs(cos_alpha[2]);
	a2 = acos(std::abs(val2)) * CONST_180_PI;

	alpha = p[0] * a0 + p[1] * a1 + p[2] * a2;

	result.push_back(entropy);
	result.push_back(anisotropy);
	result.push_back(alpha);
}

//reference
//https://github.com/senbox-org/s1tbx/blob/master/rstb/rstb-op-polarimetric-tools/src/main/java/org/csa/rstb/polarimetric/gpf/decompositions/Yamaguchi.java
void polsar::yamaguchi4Decomp(const Mat_<Complexd>& coherence, const Mat_<Complexd>& covariance, vector<double>& result) {
	double ratio, d, cR, cI, c0, s, pd, pv, ps, pc, span, k1, k2, k3;

	span = coherence.at<Complexd>(0, 0).re + coherence.at<Complexd>(1, 1).re + coherence.at<Complexd>(2, 2).re;
	pc = 2.0 * std::abs(coherence.at<Complexd>(1,2).im);
	ratio = 10.0 * std::log10(covariance.at<Complexd>(2,2).re / covariance.at<Complexd>(0,0).re);

	if (ratio <= -2.0) {
		k1 = 1.0 / 6.0;
		k2 = 7.0 / 30.0;
		k3 = 4.0 / 15.0;
	}
	else if (ratio > 2.0) {
		k1 = -1.0 / 6.0;
		k2 = 7.0 / 30.0;
		k3 = 4.0 / 15.0;
	}
	else { // -2 < ratio <= 2
		k1 = 0.0;
		k2 = 1.0 / 4.0;
		k3 = 1.0 / 4.0;
	}

	pv = (coherence.at<Complexd>(2,2).re - 0.5 * pc) / k3;

	if (pv <= m_Epsilon) { // Freeman-Durden 3 component decomposition
		pc = 0.0;
		freemanDurdenDecomp(covariance, result);
		result.push_back(pc);
	}
	else { // Yamaguchi 4 component decomposition

		s = coherence.at<Complexd>(0,0).re - 0.5 * pv;
		d = coherence.at<Complexd>(1,1).re - k2 * pv - 0.5 * pc;
		cR = coherence.at<Complexd>(0, 1).re - k1 * pv;
		cI = coherence.at<Complexd>(0, 1).im;

		if (pv + pc < span) {

			c0 = covariance.at<Complexd>(0,2).re - 0.5 * covariance.at<Complexd>(1, 1).re + 0.5 * pc;
			if (c0 < m_Epsilon) {
				ps = s - (cR * cR + cI * cI) / d;
				pd = d + (cR * cR + cI * cI) / d;
			}
			else {
				ps = s + (cR * cR + cI * cI) / s;
				pd = d - (cR * cR + cI * cI) / s;
			}

			if (ps > m_Epsilon && pd < m_Epsilon) {
				pd = 0.0;
				ps = span - pv - pc;
			}
			else if (ps < m_Epsilon && pd > m_Epsilon) {
				ps = 0.0;
				pd = span - pv - pc;
			}
			else if (ps < m_Epsilon && pd < m_Epsilon) {
				ps = 0.0;
				pd = 0.0;
				pv = span - pc;
			}

		}
		else {
			ps = 0.0;
			pd = 0.0;
			pv = span - pc;
		}
		result.push_back(ps);
		result.push_back(pd);
		result.push_back(pv);
		result.push_back(pc);
	}
}

void polsar::GetFreemanDurdenDecomp(const vector<Mat>& upcorner_covariance, vector<Mat>& decomposition) {
	int rows = upcorner_covariance[0].rows;
	int cols = upcorner_covariance[0].cols;

	// record the result
	Mat Ps = Mat::zeros(rows, cols, CV_64FC1);
	Mat Pd = Mat::zeros(rows, cols, CV_64FC1);
	Mat Pv = Mat::zeros(rows, cols, CV_64FC1);

	Mat_<Complexd>  m = Mat_<Complexd>(3, 3);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			m = restoreMatrix(upcorner_covariance, i, j);
			vector<double> result;
			freemanDurdenDecomp(m, result);
			Ps.at<double>(i, j) = result[0];
			Pd.at<double>(i, j) = result[1];
			Pv.at<double>(i, j) = result[2];
		}
	}
	decomposition.push_back(Ps);
	decomposition.push_back(Pd);
	decomposition.push_back(Pv);
}

void polsar::GetCloudePottierDecomp(const vector<Mat>& upcorner_coherence, vector<Mat>& decomposition) {
	// restore the original coherecy matrix from the diagonal and the upper elements 
	int rows = upcorner_coherence[0].rows;
	int cols = upcorner_coherence[0].cols;

	// record the result
	Mat H = Mat::zeros(rows,cols, CV_64FC1);
	Mat A = Mat::zeros(rows, cols,  CV_64FC1);
	Mat Alpha = Mat::zeros(rows, cols,  CV_64FC1);

	Mat_<Complexd> t = Mat_<Complexd>(3, 3);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j <cols; j++) {
			t = restoreMatrix(upcorner_coherence, i, j);
			vector<double> result;
			cloudePottierDecomp(t, result);
			H.at<double>(i, j) = result[0];
			Alpha.at<double>(i, j) = result[1];
			A.at<double>(i, j) = result[2];
		}
	}
	decomposition.push_back(H);
	decomposition.push_back(Alpha);
	decomposition.push_back(A);
}

void polsar::GetYamaguchi4Decomp(const vector<Mat>& upcorner_coherence, const vector<Mat> & upcorner_covariance, vector<Mat>& decomposition) {
	 
	int rows = upcorner_coherence[0].rows;
	int cols = upcorner_coherence[0].cols;
	// record the result
	Mat Ps = Mat::zeros(rows,cols, CV_64FC1);
	Mat Pd= Mat::zeros(rows, cols, CV_64FC1);
	Mat Pv = Mat::zeros(rows, cols, CV_64FC1);
	Mat Pc = Mat::zeros(rows, cols, CV_64FC1);

	// restore the original coherecy matrix from the diagonal and the upper elements 
	Mat_<Complexd> t = Mat_<Complexd>(3, 3);
	Mat_<Complexd> c = Mat_<Complexd>(3, 3);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			t = restoreMatrix(upcorner_coherence, i, j);
			c = restoreMatrix(upcorner_covariance, i, j);
			vector<double> result;
			yamaguchi4Decomp(t, c, result);
			Ps.at<double>(i, j) = result[0];
			Pd.at<double>(i, j) = result[1];
			Pv.at<double>(i, j) = result[2];
			Pc.at<double>(i, j) = result[3];
		}
	}
	decomposition.push_back(Ps);
	decomposition.push_back(Pd);
	decomposition.push_back(Pv);
	decomposition.push_back(Pc);
}


void polsar::GetCoherencyT(const vector<Mat>& pauli, vector<Mat>& upcorner_coherence, int winSize) { vec2mat(pauli, upcorner_coherence, winSize); }

void polsar::GetCovarianceC(const vector<Mat>& lexi, vector<Mat>& upcorner_covariance, int winSize) { vec2mat(lexi, upcorner_covariance, winSize); }


void polsar::GetFullPolStat(Mat& hh, const Mat& vv, const Mat& hv, vector<Mat>& result) {
	vector<Mat> data = { hh, vv, hv };
	getStatisticFeature(data, result);
}

// get statistical (min,max,mean,median,std) on polsar parameters
// vector<mat>& result -  mat size: 1*5
void polsar::getStatisticFeature(const vector<Mat>& data, vector<Mat>& result) {
	Mat hh, vv, hv, vh;
	Mat hh_log, vv_log, hv_log, vh_log;
	if (data.size() == 3) {
		hh = data[0];
		vv = data[1];
		hv = data[2];
	}
	else if (data.size() == 2) {
		vv = data[0];
		vh = data[1];
	}
	vector<Mat> temp;
	// intensity of HH channel
	if (!hh.empty()){
		 hh_log = polsar::logTransform(polsar::getComplexAmpl(hh));
		temp.push_back(hh_log);
     }
	// intensity of VV channel
	if(!vv.empty()){
	 vv_log = polsar::logTransform(polsar::getComplexAmpl(vv));
	temp.push_back(vv_log);
	}
	// intensity of HV channel
	if (!hv.empty()) {
		 hv_log = polsar::logTransform(polsar::getComplexAmpl(hv));
		temp.push_back(hv_log);
	}
	// intensity of VH channel
	if (!vh.empty()) {
		 vh_log = polsar::logTransform(polsar::getComplexAmpl(vh));
		temp.push_back(vh_log);
	}
	Mat phaseDiff;
	if(!hh.empty() && !vv.empty()){
		// phase difference HH-VV
		phaseDiff = polsar::getPhaseDiff(hh, vv);
		temp.push_back(phaseDiff);
	}

	if (!vh.empty() && !vv.empty()) {
		phaseDiff = polsar::getPhaseDiff(vv, vh);
		temp.push_back(phaseDiff);
	}

	//statistic of Co-polarize ratio VV-HH
	if(!vv_log.empty() && !hh_log.empty()){
		Mat coPolarize = vv_log - hh_log;
		temp.push_back(coPolarize);
	}
	// Cross-polarized ratio HV-HH
	if (!hv_log.empty() && !hh_log.empty()) {
		Mat crossPolarize = hv_log - hh_log;
		temp.push_back(crossPolarize);
	}
	// polarized ratio VH-VV
	if (!vh_log.empty() && !vv_log.empty()) {
		Mat otherPolarize = vh_log - vv_log;
		temp.push_back(otherPolarize);
	}

	for (const auto& t : temp) {
		result.push_back(cvFeatures::GetStatistic(t));
	}
}


// get the log upper triangle elements of matrix elements of C, T
void polsar::GetCTelements(const Mat& hh, const Mat& vv, const Mat& hv, vector<Mat>& result) {

	vector<Mat> pauli;
	vector<Mat> circ;
	vector<Mat> lexi;
	polsar::getPauliBasis(hh, vv, hv, pauli);
	polsar::getCircBasis(hh, vv, hv, circ);
	polsar::getLexiBasis(hh, vv, hv, lexi);
	vector<Mat> covariance;
	vector<Mat> coherence;
	polsar::GetCoherencyT(pauli, coherence);
	polsar::GetCovarianceC(lexi, covariance);

	// upper triangle matrix elements of covariance matrix C and coherence matrix T
	copy(coherence.begin(), coherence.end(), std::back_inserter(result));
	copy(covariance.begin(), covariance.end(), std::back_inserter(result));

	for (auto& e : result) {
		Mat temp;
		if (e.channels() == 1) { 
			temp = polsar::logTransform(e);
		}
		else if (e.channels() ==2 ) {
			temp = polsar::logTransform(polsar::getComplexAmpl(e));
		}
		e = temp;
	}
}

//restore 3*3 covariance or coherence mat from upcorner elements
Mat polsar::restoreMatrix(const vector<Mat>& mat, int row, int col) {
	Mat_<Complexd> m(3, 3);
	m.at<Complexd>(0, 0) = Complex(mat[0].at<double>(row, col), 0.0);
	m.at<Complexd>(0, 1) = Complex(mat[1].at<Vec2d>(row, col)[0], mat[1].at<Vec2d>(row, col)[1]);
	m.at<Complexd>(0, 2) = Complex(mat[2].at<Vec2d>(row, col)[0], mat[2].at<Vec2d>(row, col)[1]);
	m.at<Complexd>(1, 0) = m.at<Complexd>(0, 1).conj();
	m.at<Complexd>(1, 1) = Complex(mat[3].at<double>(row, col), 0.0);
	m.at<Complexd>(1, 2) = Complex(mat[4].at<Vec2d>(row, col)[0], mat[4].at<Vec2d>(row, col)[1]);
	m.at<Complexd>(2, 0) = m.at<Complexd>(0, 2).conj();
	m.at<Complexd>(2, 1) = m.at<Complexd>(1, 2).conj();
	m.at<Complexd>(2, 2) = Complex(mat[5].at<double>(row, col), 0.0);
	return m;
}

 