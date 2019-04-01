#ifndef PROANDFUSION_H
#define PROANDFUSION_H

#include "Frame.h"

void propogate_depth(const FramePath &frame_path_pre, const FramePath &frame_path_curr, const cv::Mat& Ki, Frame &frame_pre, Frame &frame_curr,
                     bool do_post_process);


void fuse_depth(Frame &frame, float white_noise);

#endif