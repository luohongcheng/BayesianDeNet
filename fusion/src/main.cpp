#include "frame.h"
#include "file_io.h"
#include "constant.h"
#include "pro_and_fusion.h"

int main() {

	cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
	K.at<float>(0, 0) = camera_fx;
	K.at<float>(1, 1) = camera_fy;
	K.at<float>(0, 2) = camera_cx;
	K.at<float>(1, 2) = camera_cy;

	std::vector<FramePath> frame_paths;
	//process frames

	int fusion_interval = 10;

	if (frame_paths.size() < fusion_interval)
		return -1;

	Frame frame_pre, frame_current;
	// load the first frame
	load_frame(frame_paths[0], frame_pre, image_cols, image_rows, ushort_factor);

	for (size_t i = 0; i < frame_paths.size() - fusion_interval; i += fusion_interval) {

		load_frame(frame_paths[i + fusion_interval], frame_current, image_cols, image_rows, ushort_factor);

		propogate_depth(frame_paths[i], frame_paths[i + fusion_interval], K, frame_pre, frame_current, true);

		fuse_depth(frame_current, 0.0);

		frame_pre.fused_depth = frame_current.fused_depth.clone();
		frame_pre.fused_uncertainty = frame_current.fused_uncertainty.clone();
		frame_pre.t = frame_current.t;
		frame_pre.q = frame_current.q;
	}

}