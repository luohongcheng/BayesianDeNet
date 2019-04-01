#ifndef FRAME_H
#define FRAME_H

#include <opencv2/opencv.hpp>

struct FramePath {
	FramePath() {}

	std::string rgb_path;
	std::string gt_depth_path;
	std::string pred_depth_path;
	std::string pred_confi_path;

	std::string pose_path;
};


struct Frame {
	Frame() {};

	cv::Mat rgb_image;

	//resized gt depth
	cv::Mat gt_depth;
	//raw depth is used for evalution
	cv::Mat gt_depth_raw;

	cv::Mat observed_depth;
	cv::Mat propogated_depth;
	cv::Mat fused_depth;

	cv::Mat observed_uncertainty;
	cv::Mat propogated_uncertainty;
	cv::Mat fused_uncertainty;

	std::vector<float> t; //t_x t_y t_z
	std::vector<float> q; //q_x q_y q_z q_w
};

#endif