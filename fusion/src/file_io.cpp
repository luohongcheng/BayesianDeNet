#include <fstream>

#include "file_io.h"
#include "utility.h"

void load_frame(const FramePath& frame_path, Frame& frame, int cols, int rows, float ushort_factor) {

	cv::Mat gt_depth = cv::imread(frame_path.gt_depth_path, -1);

	cv::Mat gt_depth_float = ushort_to_float(gt_depth, ushort_factor);
	frame.gt_depth_raw = gt_depth_float.clone();

	gt_depth_float = resize_depth(gt_depth_float, cols, rows);
	frame.gt_depth = gt_depth_float.clone();

	//load predicted depth
	cv::Mat obs_depth = cv::imread(frame_path.pred_depth_path, -1);
	obs_depth = ushort_to_float(obs_depth, ushort_factor);
	cv::resize(obs_depth, obs_depth, cv::Size(cols, rows));
	frame.observed_depth = obs_depth.clone();
	frame.fused_depth = obs_depth.clone();

	//load predicted confidence and convert to uncertainty
	cv::Mat obs_confidence = cv::imread(frame_path.pred_confi_path, -1);
	obs_confidence = ushort_to_float(obs_confidence, ushort_factor);
	cv::resize(obs_confidence, obs_confidence, cv::Size(cols, rows));
	cv::Mat obs_uncertainty = confidence_to_uncertainty(obs_confidence);
	frame.observed_uncertainty = obs_uncertainty.clone();
	frame.fused_uncertainty = obs_uncertainty.clone();

	//load camera poses
	std::ifstream ifs(frame_path.pose_path);
	std::string str;
	std::vector<float> q_and_t;
	while (ifs >> str) {
		q_and_t.push_back(std::atof(str.c_str()));
	}
	ifs.close();
	for (int i = 0; i < 3; ++i) {
		frame.t.push_back(q_and_t[i]);
	}
	for (int i = 3; i < 7; ++i) {
		frame.q.push_back(q_and_t[i]);
	}

}

void process_frame_paths(std::vector<FramePath>& frame_paths) {
	//TODO
	return ;
}