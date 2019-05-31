#ifndef UTILITY_H
#define UTILITY_H

#include "frame.h"

cv::Mat resize_depth(const cv::Mat &src, int cols, int rows);

cv::Mat confidence_to_uncertainty(const cv::Mat &confidence);

cv::Mat ushort_to_float(const cv::Mat &ushort_map, float ushort_factor);

#endif