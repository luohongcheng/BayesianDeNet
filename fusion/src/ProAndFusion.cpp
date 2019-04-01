#include <Eigen/Core>
#include <Eigen/Geometry>

#include "ProAndFusion.h"

void post_process(cv::Mat& propogated_depth, cv::Mat &propogated_uncertainty) {

	cv::Mat propogate_depth_temp = propogated_depth.clone();
	cv::Mat propogated_uncertainty_temp = propogated_uncertainty.clone();
	int rows = propogated_depth.rows;
	int cols = propogated_depth.cols;
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			if (propogated_depth.at<float>(i, j) > 1e-9)
				continue;
			else {

				int count = 0;
				float total = 0;
				float total_uncer = 0;
				int window_size = 3;

				for (int ii = i - window_size <= 0 ? 0 : i - window_size; ii <= (i + window_size >= rows - 1 ? rows - 1 : i + window_size); ++ii) {
					for (int jj = j - window_size <= 0 ? 0 : j - window_size; jj <= (j + window_size >= cols - 1 ? cols - 1 : j + window_size); ++jj) {

						if (propogated_depth.at<float>(ii, jj) > 1e-6) {
							total += propogated_depth.at<float>(ii, jj);
							total_uncer += propogated_uncertainty.at<float>(ii, jj);

							count++;
						}
					}
				}
				if (count != 0) {
					propogate_depth_temp.at<float>(i, j) = total / (count);
					propogated_uncertainty_temp.at<float>(i, j) = total_uncer / (count);
				}

			}

		}
	}
	propogated_depth = propogate_depth_temp.clone();
	propogated_uncertainty = propogated_uncertainty_temp.clone();
}

void propogate_depth(const FramePath &frame_path_pre, const FramePath &frame_path_curr, const cv::Mat& Ki, Frame &frame_pre, Frame &frame_curr,
                     bool do_post_process) {


	float camera_fx = Ki.at<float>(0, 0);
	float camera_fy = Ki.at<float>(1, 1);
	float camera_cx = Ki.at<float>(0, 2);
	float camera_cy = Ki.at<float>(1, 2);

	Eigen::Matrix3d K;
	K << camera_fx, 0, camera_cx, 0, camera_fy, camera_cy, 0, 0, 1;

	Eigen::Quaterniond q1 = Eigen::Quaterniond(frame_pre.q[3], frame_pre.q[0], frame_pre.q[1], frame_pre.q[2]);
	Eigen::Quaterniond q2 = Eigen::Quaterniond(frame_curr.q[3], frame_curr.q[0], frame_curr.q[1], frame_curr.q[2]);
	Eigen::Matrix3d r1 = Eigen::Matrix3d::Identity();
	r1 = q1.toRotationMatrix();
	Eigen::Matrix3d r2 = Eigen::Matrix3d::Identity();
	r2 = q2.toRotationMatrix();
	Eigen::Matrix3d r = r2.inverse() * r1;
	cv::Mat rr = (cv::Mat_<double>(3, 3) <<
	              r(0, 0), r(0, 1), r(0, 2),
	              r(1, 0), r(1, 1), r(1, 2),
	              r(2, 0), r(2, 1), r(2, 2));
	Eigen::Vector3d t1;
	t1 << frame_pre.t[0], frame_pre.t[1], frame_pre.t[2];
	Eigen::Vector3d t2;
	t2 << frame_curr.t[0], frame_curr.t[1], frame_curr.t[2];
	Eigen::Vector3d t = r2.inverse() * (t1 - t2);

	float absolute_scale = 1.0;
	cv::Mat propogated_depth = cv::Mat::zeros(frame_pre.fused_depth.size(), CV_32FC1);
	cv::Mat propogated_uncertainty = cv::Mat::zeros(frame_pre.fused_depth.size(), CV_32FC1);
	int cols = propogated_uncertainty.cols;
	int rows = propogated_depth.rows;

	for (int i = 0; i < propogated_depth.rows; ++i) {
		for (int j = 0; j < propogated_depth.cols; ++j) {
			float d = frame_pre.fused_depth.at<float>(i, j);
			if (d < 1e-9)
				continue;

			Eigen::Vector3d points_in_3d;
			double z = d;
			double x = (j - camera_cx) * z / camera_fx;
			double y = (i - camera_cy) * z / camera_fy;
			points_in_3d << x, y, z;

			points_in_3d = points_in_3d;

			Eigen::Vector3d normalized = K * (r * points_in_3d + t * absolute_scale);

			if (normalized[2] <= 0)
				continue;

			float u = normalized[0] / normalized[2];
			float v = normalized[1] / normalized[2];

			if (u < cols && u >= 0 && v < rows && v >= 0) {
				//warp_image.at<Vec3b>(v, u) = rgb1.at<Vec3b>(i, j);
				propogated_depth.at<float>(v, u) = normalized[2];
				propogated_uncertainty.at<float>(v, u) = frame_pre.fused_uncertainty.at<float>(i, j);
			}
		}
	}
	if (do_post_process)
		post_process(propogated_depth, propogated_uncertainty);
	frame_curr.propogated_depth = propogated_depth.clone();
	frame_curr.propogated_uncertainty = propogated_uncertainty.clone();

}

void fuse_depth(Frame &frame, float white_noise) {

	cv::Mat fused_depth = cv::Mat::zeros(frame.observed_depth.size(), CV_32FC1);
	cv::Mat fused_uncertainty = cv::Mat::zeros(frame.observed_uncertainty.size(), CV_32FC1);
	int rows = frame.observed_depth.rows;
	int cols = frame.observed_depth.cols;

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {

			float pro_depth_value = frame.propogated_depth.at<float>(i, j);
			float obs_depth_value = frame.observed_depth.at<float>(i, j);
			float pro_uncertainty_value = frame.propogated_uncertainty.at<float>(i, j);
			float obs_uncertrainty_value = frame.observed_uncertainty.at<float>(i, j);

			float fused_depth_value;
			float fused_uncertainty_value;

			if (pro_depth_value < 1e-5 || pro_uncertainty_value < 1e-9) {
				fused_depth_value = obs_depth_value;
				fused_uncertainty_value = obs_uncertrainty_value;
			}
			else {
				pro_uncertainty_value += white_noise;
				fused_depth_value
				    = (pro_depth_value * obs_uncertrainty_value + obs_depth_value * pro_uncertainty_value) / (pro_uncertainty_value + obs_uncertrainty_value);
				fused_uncertainty_value
				    = (obs_uncertrainty_value * pro_uncertainty_value) / (pro_uncertainty_value + obs_uncertrainty_value);
			}

			fused_depth.at<float>(i, j) = fused_depth_value;
			fused_uncertainty.at<float>(i, j) = fused_uncertainty_value;
		}
	}

	frame.fused_depth = fused_depth.clone();
	frame.fused_uncertainty = fused_uncertainty.clone();
}
