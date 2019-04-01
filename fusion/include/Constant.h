#ifndef CONSTANT_H
#define CONSTANT_H
#include <string>

const float ushort_factor = 5000.0;
const float image_scale = 2.0;

const float camera_fx = 500 / image_scale;
const float camera_fy = 500 / image_scale;
const float camera_cx = 320 / image_scale;
const float camera_cy = 240 / image_scale;

const int image_cols = 640 / image_scale;
const int image_rows = 480 / image_scale;

#endif