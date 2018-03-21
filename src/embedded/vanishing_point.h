//
// Created by Perceval Wajsburt on 20/03/2018.
//

#ifndef FPGAROADDETECT_EXTRACT_VANISHING_HPP
#define FPGAROADDETECT_EXTRACT_VANISHING_HPP

#define HOUGH_LINES_COUNT 100
#define PARALLEL_Y -9999
#define TOP_LINES 5

#include "hls_video.h"
typedef ap_axiu<24,1,1,1> interface_t;
typedef hls::stream<interface_t> stream_t;
#define MAX_HEIGHT 360
#define MAX_WIDTH 490
typedef hls::Mat<MAX_HEIGHT, MAX_WIDTH, HLS_8UC3> rgb_img_t;
typedef hls::Mat<MAX_HEIGHT, MAX_WIDTH, HLS_8UC1> gray_img_t;

typedef hls::Scalar<2, float> Vec2f;
typedef hls::Scalar<3, float> Vec3f;

typedef struct {
  int edge_threshold;
  float theta_limit;
  int hough_treshold;
  int blur_size;
  int mask_mid_y;
  int mask_mid_value;
  bool display_extra;
  bool display_raw;
} Params;

typedef struct {
  float mean_x;
  float mean_y;
} Output;

hls::Mat mask(MAX_HEIGHT, MAX_WIDTH, HLS_32FC1);
bool mask_initialized = false;

Vec2f intersection(Vec3f l1, Vec3f l2);
void extract_vanishing(stream_t& stream_in, stream_t& stream_out, Params * params, Output * output);


#endif //FPGAROADDETECT_EXTRACT_VANISHING_HPP