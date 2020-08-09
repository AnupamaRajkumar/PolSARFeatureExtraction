#include "sarFeatures.hpp"

using namespace cv;
using namespace std;

static constexpr float m_Epsilon = 1e-13f;
const float PI_F = 3.14159265358979f;
const float CONST_180_PI = 180.0f / PI_F;

Mat polsar::getComplexAmpl(const Mat& in) {

	Mat out, phase;
	vector<Mat> channels;
	split(in, channels);
	cv::cartToPolar(channels[0], channels[1], out, phase);
	
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
	pauli.push_back((hh + vv) / sqrt(2.0));
	pauli.push_back((hh - vv) / sqrt(2.0));
	pauli.push_back(hv * sqrt(2.0));
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

// return the log form
Mat polsar::getCoPolarizationRatio(const Mat& hh, const Mat& vv, int winSize) {
	Mat output1, output2, output;
	output1 = calcuCoherenceOfPol(vv, vv, winSize);
	output2 = calcuCoherenceOfPol(hh, hh, winSize);

	output =logTransform(output1) - logTransform(output2);
	return output;
}

// return the log form
Mat polsar::getDePolarizationRatio(const Mat& hh, const Mat& vv, const Mat& hv, int winSize) {

	Mat output;
	Mat output1 = calcuCoherenceOfPol(hv, hv, winSize);
	Mat	output2 = calcuCoherenceOfPol(hh, hh, winSize);
	Mat output3 = calcuCoherenceOfPol(vv, vv, winSize);  

	output = logTransform(output1) - logTransform(output2 + output3);
	return output;
}

// get <band1*conju(band2)>
// output: 2 channel matrix
Mat polsar::calcuCoherenceOfPol(const Mat& band1, const Mat& band2, int winSize) {
	Mat output1, output;
	cv::mulSpectrums(band1, band2, output1, 0, true); // band1* conju(band2)
	cv::blur(output1, output, Size(winSize, winSize));
	return output;
}

// get the relative phases
Mat polsar::getPhaseDiff(const Mat& hh, const Mat& vv) {
	//Mat temp = getMul(hh, getConj(vv));
	//return getComplexAngle(temp);
	Mat hh_real, hh_imag, vv_real, vv_imag;
	extractChannel( hh, hh_real, 0);
	extractChannel( hh, hh_imag, 1);
	extractChannel( vv, vv_real, 0);
	extractChannel(vv, vv_imag, 1);
	Mat output= Mat(Size(hh.size()), CV_32FC1);
	for (int i = 0; i < hh.rows; i++) {
		for (int j = 0; j < hh.cols; j++) {
			output.at<float>(i, j) = atan(hh_imag.at<float>(i, j) / hh_real.at<float>(i, j)) - atan(vv_imag.at<float>(i, j) / vv_real.at<float>(i, j));
		}
	}
	return output;
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
	
	Mat R = logTransform(getComplexAmpl(hh+vv));
	Mat G = logTransform(getComplexAmpl(hv));
	Mat B = logTransform(getComplexAmpl(hh-vv));

	////cut everything over 2.5x the mean value
	//float R_mean = cv::mean(R)[0];
	//float G_mean = cv::mean(G)[0];
	//float B_mean = cv::mean(B)[0];
	//threshold(R, R, 2.5 * R_mean, 2.5 * R_mean, THRESH_TRUNC);
	//threshold(G, G, 2.5 * G_mean, 2.5 * G_mean, THRESH_TRUNC);
	//threshold(B, B, 2.5 * B_mean, 2.5 * B_mean, THRESH_TRUNC);
	return GetColorImg(R, G, B, true);
}

// get the complex conjugation of a 2 channel matrix
Mat polsar::getConj(const Mat& src) {
	Mat temp = Mat(Size(src.size()), CV_32FC2);
	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++) {
			temp.at<Vec2f>(i, j)[0] = src.at<Vec2f>(i, j)[0];
			temp.at<Vec2f>(i, j)[1] = src.at<Vec2f>(i, j)[1] * -1.0;
		}
	return temp;
}

// get the complex muliplication of two 2 channel matrix
Mat polsar::getMul(const Mat& src1, const Mat& src2) {
	Mat temp = Mat(Size(src1.size()), CV_32FC2);
	for (int i = 0; i < src1.rows; i++)
		for (int j = 0; j < src1.cols; j++) {
			// (a+bi)(c+di) = ac-bd + (ad + bc)i
			float a = src1.at<Vec2f>(i, j)[0];
			float b = src1.at<Vec2f>(i, j)[1];
			float c = src2.at<Vec2f>(i, j)[0];
			float d = src2.at<Vec2f>(i, j)[1];
			temp.at<Vec2f>(i, j)[0] = a * c - b * d;
			temp.at<Vec2f>(i, j)[1] = a * d + b * c;
		}
	return temp;
}



// convert CV_32FC2 to Complexf
Mat_<Complexf> polsar::getComplexMat(const Mat& src) {
	Mat dst = Mat_<Complexf>(Size(src.size()));
	if (src.channels() == 2) {
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				dst.at<Complexf>(i, j).re = src.at<Vec2f>(i, j)[0];
				dst.at<Complexf>(i, j).im = src.at<Vec2f>(i, j)[1];
			}
		}
	}
	else if (src.channels() == 1) {
		for (int i = 0; i < src.rows; i++) {
			for (int j = 0; j < src.cols; j++) {
				dst.at<Complexf>(i, j).re = src.at<float>(i, j);
				dst.at<Complexf>(i, j).im = 0.0f;
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
	k_d = Mat(Size(s_ll_amp.size()), s_ll_amp.type());
	for (int i = 0; i < s_ll_amp.rows; i++) {
		for (int j = 0; j < s_ll_amp.cols; j++) {
			k_d.at<float>(i, j) = std::min(s_ll_amp.at<float>(i, j), s_rr_amp.at<float>(i, j));
		}
	}
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


// reference
// https://github.com/senbox-org/s1tbx/blob/master/rstb/rstb-op-polarimetric-tools/src/main/java/org/csa/rstb/polarimetric/gpf/decompositions/FreemanDurden.java
void polsar::freemanDurdenDecomp(const Mat_<Complexf>& covariance, vector<float>& result) {

	float fd, fv, fs, c11, c13Re, c13Im, c33, alphaRe, alphaIm, betaRe, betaIm;
	// compute fv from m22 and subtract fv from m11, m13, m33
	fv = 4.0f * covariance.at<Complexf>(1, 1).re;
	c11 = covariance.at<Complexf>(0, 0).re - fv * 3.0f / 8.0f;
	c13Re = covariance.at<Complexf>(0, 2).re - fv / 8.0f;
	c13Im = covariance.at<Complexf>(0, 2).im;
	c33 = covariance.at<Complexf>(2, 2).re - fv * 3.0f / 8.0f;
	float a1 = c11 * c33;

	if (c11 <= m_Epsilon || c33 <= m_Epsilon) {
		fs = 0.0f;
		fd = 0.0f;
		alphaRe = 0.0f;
		alphaIm = 0.0f;
		betaRe = 0.0f;
		betaIm = 0.0f;
	}
	else {

		float a2 = c13Re * c13Re + c13Im * c13Im;
		if (a1 < a2) {
			float c13 = std::sqrt(a2);
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
	float ps = fs * (1 + betaRe * betaRe + betaIm * betaIm);
	float pd = fd * (1 + alphaRe * alphaRe + alphaIm * alphaIm);
	float pv = fv;
	result.push_back(ps);
	result.push_back(pd);
	result.push_back(pv);
}


// reference:
// https://github.com/senbox-org/s1tbx/blob/master/rstb/rstb-op-polarimetric-tools/src/main/java/org/csa/rstb/polarimetric/gpf/decompositions/hAAlpha.java
void polsar::cloudePottierDecomp(Mat_<Complexf>& coherence, vector<float>& result) {

	Mat HMr = Mat(3, 3, CV_32FC1); // the real part of Hermitian matrix
	Mat HMi = Mat(3, 3, CV_32FC1); // the imag part of Hermitian matrix
	for (int i = 0; i < coherence.rows; i++) {
		for (int j = 0; j < coherence.cols; j++) {
			HMr.at<float>(i, j) = coherence.at<Complexf>(i, j).re;
			HMi.at<float>(i, j) = coherence.at<Complexf>(i, j).im;
		}
	}

	Mat EigenVectRe = Mat(3,3, CV_32FC1);
	Mat EigenVectIm = Mat(3, 3, CV_32FC1);
	vector<float> EigenVal(3);
	std::array<float,3> lambda , p, alpha, phi, beta, delta, gamma;
	eigenDecomposition(3, HMr, HMi, EigenVectRe, EigenVectIm, EigenVal);
	float sum = 0.0f;
	for (int i = 0; i < 3; ++i) {
		lambda[i] = EigenVal[i];
		sum += lambda[i];
	}

	float EPS = m_Epsilon;
	for (int j = 0; j < 3; ++j) {
		alpha[j] = std::acos(std::sqrt(EigenVectRe.at<float>(0,j)* EigenVectRe.at<float>(0, j)+ EigenVectIm.at<float>(0, j)*EigenVectIm.at<float>(0,j))) * CONST_180_PI;
		beta[j] = std::atan2(std::sqrt(EigenVectRe.at<float>(2,j)* EigenVectRe.at<float>(2, j)+ EigenVectIm.at<float>(2, j)*EigenVectIm.at<float>(2,j)),
			EPS + std::sqrt(EigenVectRe.at<float>(1,j)* EigenVectRe.at<float>(1, j)+EigenVectIm.at<float>(1,j)* EigenVectIm.at<float>(1, j))) * CONST_180_PI;
		phi[j] = std::atan2(EigenVectIm.at<float>(0,j), EPS + EigenVectRe.at<float>(0,j));
		delta[j] = std::atan2(EigenVectIm.at<float>(1,j), EPS + EigenVectRe.at<float>(1,j)) - phi[j];
		delta[j] = std::atan2(std::sin(delta[j]), std::cos(delta[j]) + EPS) * CONST_180_PI;
		gamma[j] = std::atan2(EigenVectIm.at<float>(2,j), EPS + EigenVectRe.at<float>(2,j)) - phi[j];
		gamma[j] = std::atan2(std::sin(gamma[j]),std::cos(gamma[j]) + EPS) * CONST_180_PI;
		p[j] = lambda[j] / sum;
		if (p[j] < 0) {
			p[j] = 0;
		}
		else if (p[j] > 1) {
			p[j] = 1;
		}
	}

	float meanLambda = 0.0f;
	float meanAlpha = 0.0f;
	float meanBeta = 0.0f;
	float meanDelta = 0.0f;
	float meanGamma = 0.0f;
	float entropy = 0.0f;
	for (int k = 0; k < 3; ++k) {
		meanLambda += p[k] * lambda[k];
		meanAlpha += p[k] * alpha[k];
		meanBeta += p[k] * beta[k];
		meanDelta += p[k] * delta[k];
		meanGamma += p[k] * gamma[k];
		entropy -= p[k] * std::log10(p[k] + EPS);
	}

	entropy /= std::log10(3.0f);
	float anisotropy = (p[1] - p[2]) / (p[1] + p[2] + EPS);

	result.push_back(entropy);
	result.push_back(anisotropy);
	//result.push_back(meanAlpha);
	//result.push_back(meanBeta);
	//result.push_back(meanDelta);
	//result.push_back(meanGamma);
	//result.push_back(meanLambda);
	result.push_back(alpha[0]);
	result.push_back(alpha[1]); 
	result.push_back(alpha[2]);
	//result.push_back(lambda[0]);
	//result.push_back(lambda[1]);
	//result.push_back(lambda[2]);
}

 

//reference
//https://github.com/senbox-org/s1tbx/blob/master/rstb/rstb-op-polarimetric-tools/src/main/java/org/csa/rstb/polarimetric/gpf/decompositions/Yamaguchi.java
void polsar::yamaguchi4Decomp(const Mat_<Complexf>& coherence, const Mat_<Complexf>& covariance, vector<float>& result) {
	float ratio, d, cR, cI, c0, s, pd, pv, ps, pc, span, k1, k2, k3;

	span = coherence.at<Complexf>(0, 0).re + coherence.at<Complexf>(1, 1).re + coherence.at<Complexf>(2, 2).re;
	pc = 2.0f * std::abs(coherence.at<Complexf>(1,2).im);
	ratio = 10.0f * std::log10(covariance.at<Complexf>(2,2).re / covariance.at<Complexf>(0,0).re);

	if (ratio <= -2.0f) {
		k1 = 1.0f / 6.0f;
		k2 = 7.0f / 30.0f;
		k3 = 4.0f / 15.0f;
	}
	else if (ratio > 2.0f) {
		k1 = -1.0f / 6.0f;
		k2 = 7.0f / 30.0f;
		k3 = 4.0f / 15.0f;
	}
	else { // -2 < ratio <= 2
		k1 = 0.0f;
		k2 = 1.0f / 4.0f;
		k3 = 1.0f / 4.0f;
	}

	pv = (coherence.at<Complexf>(2,2).re - 0.5f * pc) / k3;

	if (pv <= m_Epsilon) { // Freeman-Durden 3 component decomposition
		pc = 0.0f;
		freemanDurdenDecomp(covariance, result);
		result.push_back(pc);
	}
	else { // Yamaguchi 4 component decomposition

		s = coherence.at<Complexf>(0,0).re - 0.5f * pv;
		d = coherence.at<Complexf>(1,1).re - k2 * pv - 0.5f * pc;
		cR = coherence.at<Complexf>(0, 1).re - k1 * pv;
		cI = coherence.at<Complexf>(0, 1).im;

		if (pv + pc < span) {

			c0 = covariance.at<Complexf>(0,2).re - 0.5f * covariance.at<Complexf>(1, 1).re + 0.5f * pc;
			if (c0 < m_Epsilon) {
				ps = s - (cR * cR + cI * cI) / d;
				pd = d + (cR * cR + cI * cI) / d;
			}
			else {
				ps = s + (cR * cR + cI * cI) / s;
				pd = d - (cR * cR + cI * cI) / s;
			}

			if (ps > m_Epsilon && pd < m_Epsilon) {
				pd = 0.0f;
				ps = span - pv - pc;
			}
			else if (ps < m_Epsilon && pd > m_Epsilon) {
				ps = 0.0f;
				pd = span - pv - pc;
			}
			else if (ps < m_Epsilon && pd < m_Epsilon) {
				ps = 0.0f;
				pd = 0.0f;
				pv = span - pc;
			}

		}
		else {
			ps = 0.0f;
			pd = 0.0f;
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
	Mat Ps = Mat::zeros(rows, cols, CV_32FC1);
	Mat Pd = Mat::zeros(rows, cols, CV_32FC1);
	Mat Pv = Mat::zeros(rows, cols, CV_32FC1);

	Mat_<Complexf>  m = Mat_<Complexf>(3, 3);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			m = restoreMatrix(upcorner_covariance, i, j);
			vector<float> result;
			freemanDurdenDecomp(m, result);
			Ps.at<float>(i, j) = result[0];
			Pd.at<float>(i, j) = result[1];
			Pv.at<float>(i, j) = result[2];
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

	decomposition = vector<Mat>(5);
	for (auto& d : decomposition) {
		d = Mat(rows, cols, CV_32FC1);
	}
	Mat_<Complexf> t = Mat_<Complexf>(3, 3);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j <cols; j++) {
			t = restoreMatrix(upcorner_coherence, i, j);
			vector<float> result;
			cloudePottierDecomp(t, result);
			for (int k = 0; k < result.size(); k++) {
				decomposition[k].at<float>(i, j) = result[k];
			}
		}
	}
}

void polsar::GetYamaguchi4Decomp(const vector<Mat>& upcorner_coherence, const vector<Mat> & upcorner_covariance, vector<Mat>& decomposition) {
	 
	int rows = upcorner_coherence[0].rows;
	int cols = upcorner_coherence[0].cols;
	// record the result
	Mat Ps = Mat::zeros(rows,cols, CV_32FC1);
	Mat Pd= Mat::zeros(rows, cols, CV_32FC1);
	Mat Pv = Mat::zeros(rows, cols, CV_32FC1);
	Mat Pc = Mat::zeros(rows, cols, CV_32FC1);

	// restore the original coherecy matrix from the diagonal and the upper elements 
	Mat_<Complexf> t = Mat_<Complexf>(3, 3);
	Mat_<Complexf> c = Mat_<Complexf>(3, 3);
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			t = restoreMatrix(upcorner_coherence, i, j);
			c = restoreMatrix(upcorner_covariance, i, j);
			vector<float> result;
			yamaguchi4Decomp(t, c, result);
			Ps.at<float>(i, j) = result[0];
			Pd.at<float>(i, j) = result[1];
			Pv.at<float>(i, j) = result[2];
			Pc.at<float>(i, j) = result[3];
		}
	}
	decomposition.push_back(Ps);
	decomposition.push_back(Pd);
	decomposition.push_back(Pv);
	decomposition.push_back(Pc);
}


void polsar::GetCoherencyT(const vector<Mat>& pauli, vector<Mat>& upcorner_coherence, int winSize) { vec2mat(pauli, upcorner_coherence, winSize); }

void polsar::GetCovarianceC(const vector<Mat>& lexi, vector<Mat>& upcorner_covariance, int winSize) { vec2mat(lexi, upcorner_covariance, winSize); }


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
	Mat_<Complexf> m(3, 3);
	m.at<Complexf>(0, 0) = Complex(mat[0].at<float>(row, col), 0.0f);
	m.at<Complexf>(0, 1) = Complex(mat[1].at<Vec2f>(row, col)[0], mat[1].at<Vec2f>(row, col)[1]);
	m.at<Complexf>(0, 2) = Complex(mat[2].at<Vec2f>(row, col)[0], mat[2].at<Vec2f>(row, col)[1]);
	m.at<Complexf>(1, 0) = m.at<Complexf>(0, 1).conj();
	m.at<Complexf>(1, 1) = Complex(mat[3].at<float>(row, col), 0.0f);
	m.at<Complexf>(1, 2) = Complex(mat[4].at<Vec2f>(row, col)[0], mat[4].at<Vec2f>(row, col)[1]);
	m.at<Complexf>(2, 0) = m.at<Complexf>(0, 2).conj();
	m.at<Complexf>(2, 1) = m.at<Complexf>(1, 2).conj();
	m.at<Complexf>(2, 2) = Complex(mat[5].at<float>(row, col), 0.0f);
	return m;
}


/**
	 * Perform eigenvalue decomposition for a given Hermitian matrix
	 *
	 * @param n           Matrix dimension
	 * @param HMr         the real part of the Hermitian matrix 
	 * @param HMi         the imag part of the Hermitian matrix 
	 * @param EigenVectRe Real part of the eigenvector matrix
	 * @param EigenVectIm Imaginary part of the eigenvector matrix
	 * @param EigenVal    Eigenvalue vector
	 */
void polsar::eigenDecomposition(int n, const Mat& HMr, const Mat& HMi, Mat & EigenVectRe, Mat & EigenVectIm, vector<float> & EigenVal) {

	Mat ar = Mat(n,n,CV_32FC1);
	Mat ai = Mat(n, n, CV_32FC1);
	Mat vr = Mat(n, n, CV_32FC1);
	Mat vi = Mat(n, n, CV_32FC1);
	vector<float> d(n),z(n);
	array<float,2> w ,s,c,titi,gc,hc;
	float sm, tresh, x, toto, e, f, g, h, r, d1, d2;
	int n2 = n * n;

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			ar.at<float>(i,j) = HMr.at<float>(i,j);
			ai.at<float>(i,j) = HMi.at<float>(i,j);
			vr.at<float>(i,j) = 0.0;
			vi.at<float>(i,j) = 0.0;
		}
		vr.at<float>(i, i) = 1.0f;
		vi.at<float>(i, i) = 0.0f;

		d[i] = ar.at<float>(i, i);
		z[i] = 0.0f;
	}

	 int iiMax = 1000 * n2;
	for (int ii = 1; ii < iiMax; ii++) {

		sm = 0.;
		for (int p = 0; p < n - 1; p++) {
			for (int q = p + 1; q < n; q++) {
				sm += 2.0f * std::sqrt(ar.at<float>(p,q) * ar.at<float>(p,q) + ai.at<float>(p,q) * ai.at<float>(p,q));
			}
		}
		sm /= (n2 - n);

		if (sm < 1.E-16f) {
			break;
		}

		tresh = 1.E-17f;
		if (ii < 4) {
			tresh = (long)0.2 * sm / n2;
		}

		x = -1.E-15f;
		int p = 0;
		int q = 0;
		for (int i = 0; i < n - 1; i++) {
			for (int j = i + 1; j < n; j++) {
				toto = std::sqrt(ar.at<float>(i,j) * ar.at<float>(i,j) + ai.at<float>(i,j) * ai.at<float>(i,j));
				if (x < toto) {
					x = toto;
					p = i;
					q = j;
				}
			}
		}
		toto = std::sqrt(ar.at<float>(p,q) * ar.at<float>(p,q) + ai.at<float>(p,q) * ai.at<float>(p,q));
		if (toto > tresh) {
			e = d[p] - d[q];
			w[0] = ar.at<float>(p,q);
			w[1] = ai.at<float>(p,q);
			g = std::sqrt(w[0] * w[0] + w[1] * w[1]);
			g = g * g;
			f = std::sqrt(e * e + 4.0f * g);
			d1 = e + f;
			d2 = e - f;
			if (std::abs(d2) > std::abs(d1)) {
				d1 = d2;
			}
			r = std::abs(d1) / std::sqrt(d1 * d1 + 4.0f * g);
			s[0] = r;
			s[1] = 0.0f;
			titi[0] = 2.0f * r / d1;
			titi[1] = 0.0f;
			c[0] = titi[0] * w[0] - titi[1] * w[1];
			c[1] = titi[0] * w[1] + titi[1] * w[0];
			r = std::sqrt(s[0] * s[0] + s[1] * s[1]);
			r = r * r;
			h = (d1 / 2.0f + 2.0f * g / d1) * r;
			d[p] = d[p] - h;
			z[p] = z[p] - h;
			d[q] = d[q] + h;
			z[q] = z[q] + h;
			ar.at<float>(p,q) = 0.0f;
			ai.at<float>(p,q) = 0.0f;

			for (int j = 0; j < p; j++) {
				gc[0] = ar.at<float>(j,p);
				gc[1] = ai.at<float>(j,p);
				hc[0] = ar.at<float>(j,q);
				hc[1] = ai.at<float>(j,q);
				ar.at<float>(j,p) = c[0] * gc[0] - c[1] * gc[1] - s[0] * hc[0] - s[1] * hc[1];
				ai.at<float>(j,p) = c[0] * gc[1] + c[1] * gc[0] - s[0] * hc[1] + s[1] * hc[0];
				ar.at<float>(j,q) = s[0] * gc[0] - s[1] * gc[1] + c[0] * hc[0] + c[1] * hc[1];
				ai.at<float>(j,q) = s[0] * gc[1] + s[1] * gc[0] + c[0] * hc[1] - c[1] * hc[0];
			}
			for (int j = p + 1; j < q; j++) {
				gc[0] = ar.at<float>(p,j);
				gc[1] = ai.at<float>(p,j);
				hc[0] = ar.at<float>(j,q);
				hc[1] = ai.at<float>(j,q);
				ar.at<float>(p,j) = c[0] * gc[0] + c[1] * gc[1] - s[0] * hc[0] - s[1] * hc[1];
				ai.at<float>(p,j) = c[0] * gc[1] - c[1] * gc[0] + s[0] * hc[1] - s[1] * hc[0];
				ar.at<float>(j,q) = s[0] * gc[0] + s[1] * gc[1] + c[0] * hc[0] + c[1] * hc[1];
				ai.at<float>(j,q) = -s[0] * gc[1] + s[1] * gc[0] + c[0] * hc[1] - c[1] * hc[0];
			}
			for (int j = q + 1; j < n; j++) {
				gc[0] = ar.at<float>(p,j);
				gc[1] = ai.at<float>(p,j);
				hc[0] = ar.at<float>(q,j);
				hc[1] = ai.at<float>(q,j);
				ar.at<float>(p,j) = c[0] * gc[0] + c[1] * gc[1] - s[0] * hc[0] + s[1] * hc[1];
				ai.at<float>(p,j) = c[0] * gc[1] - c[1] * gc[0] - s[0] * hc[1] - s[1] * hc[0];
				ar.at<float>(q,j) = s[0] * gc[0] + s[1] * gc[1] + c[0] * hc[0] - c[1] * hc[1];
				ai.at<float>(q,j) = s[0] * gc[1] - s[1] * gc[0] + c[0] * hc[1] + c[1] * hc[0];
			}
			for (int j = 0; j < n; j++) {
				gc[0] = vr.at<float>(j,p);
				gc[1] = vi.at<float>(j,p);
				hc[0] = vr.at<float>(j,q);
				hc[1] = vi.at<float>(j,q);
				vr.at<float>(j,p) = c[0] * gc[0] - c[1] * gc[1] - s[0] * hc[0] - s[1] * hc[1];
				vi.at<float>(j,p) = c[0] * gc[1] + c[1] * gc[0] - s[0] * hc[1] + s[1] * hc[0];
				vr.at<float>(j,q) = s[0] * gc[0] - s[1] * gc[1] + c[0] * hc[0] + c[1] * hc[1];
				vi.at<float>(j,q) = s[0] * gc[1] + s[1] * gc[0] + c[0] * hc[1] - c[1] * hc[0];
			}
		}
	}

	for (int k = 0; k < n; k++) {
		d[k] = 0;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				d[k] = d[k] + vr.at<float>(i,k) * (HMr.at<float>(i,j) * vr.at<float>(j,k) - HMi.at<float>(i,j) * vi.at<float>(j,k));
				d[k] = d[k] + vi.at<float>(i,k) * (HMr.at<float>(i,j) * vi.at<float>(j,k) + HMi.at<float>(i,j) * vr.at<float>(j,k));
			}
		}
	}

	float tmp_r, tmp_i;
	for (int i = 0; i < n; i++) {
		for (int j = i + 1; j < n; j++) {
			if (d[j] > d[i]) {
				x = d[i];
				d[i] = d[j];
				d[j] = x;
				for (int k = 0; k < n; k++) {
					tmp_r = vr.at<float>(k,i);
					tmp_i = vi.at<float>(k,i);
					vr.at<float>(k,i) = vr.at<float>(k,j);
					vi.at<float>(k,i) = vi.at<float>(k,j);
					vr.at<float>(k,j) = tmp_r;
					vi.at<float>(k,j) = tmp_i;
				}
			}
		}
	}

	if (EigenVal.size() == 0) {
		EigenVal = vector<float>(n);
	}
	if (EigenVectIm.rows == 0) {
		EigenVectIm = Mat(n, n, CV_32FC1);
	}
	if (EigenVectRe.rows == 0) {
		EigenVectRe = Mat(n, n, CV_32FC1);
	}
	for (int i = 0; i < n; i++) {
		EigenVal[i] = d[i];
		for (int j = 0; j < n; j++) {
			EigenVectRe.at<float>(i,j) = vr.at<float>(i,j);
			EigenVectIm.at<float>(i,j) = vi.at<float>(i,j);
		}
	}
}
 