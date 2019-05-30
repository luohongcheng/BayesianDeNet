#ifndef FILE_IO_H
#define FILE_IO_H

#include "frame.h"

void process_frame_paths(std::vector<FramePath>& frame_paths);

void load_frame(const FramePath& frame_path, Frame& frame, int cols, int rows, float ushort_factor);

#endif