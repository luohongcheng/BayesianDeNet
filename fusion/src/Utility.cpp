#include "utility.h"

cv::Mat resize_depth(const cv::Mat &src, int cols, int rows) {

	cv::Mat dst = cv::Mat(cv::Size(cols, rows), CV_32FC1);
	float scale = src.cols / (float)cols;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			dst.at<float>(i, j) = src.at<float>(scale * i, scale * j);
		}
	}
	return dst;
}

cv::Mat confidence_to_uncertainty(const cv::Mat &confidence) {

	cv::Mat uncertainty = cv::Mat::zeros(confidence.size(), CV_32FC1);

	for (int i = 0; i < confidence.rows; ++i) {
		for (int j = 0; j < confidence.cols; ++j) {

			float value = confidence.at<float>(i, j);
			if (value >= 1)
				value = 0.99999999;
			if (value <= 0 )
				value = 1e-6;
			uncertainty.at<float>(i, j) = pow(-log(value), 2);
		}
	}
	return uncertainty;
}

cv::Mat ushort_to_float(const cv::Mat &ushort_map, float ushort_factor) {

	cv::Mat float_map = cv::Mat::zeros(ushort_map.size(), CV_32FC1);

	for (int i = 0; i < ushort_map.rows; ++i) {
		for (int j = 0; j < ushort_map.cols; ++j) {
			float_map.at<float>(i, j) = (ushort_map.at<ushort>(i, j) / ushort_factor);
		}
	}
	return float_map;
}