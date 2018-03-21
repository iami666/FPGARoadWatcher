//
// Created by Perceval Wajsburt on 20/03/2018.
//

#include <cmath>
#include <iostream>
#include "extract_vanishing.hpp"

using namespace std;
using namespace cv;

void draw_dot(Mat &img, Vec2i center, int size) {
  for (int x = center[0] - size / 2; x < center[0] + size / 2; x++) {
    for (int y = center[1] - size / 2; y < center[1] + size / 2; y++) {
      Vec<uchar, 3> &v = img.at<Vec<uchar, 3> >(y, x);
      v[0] = 0;
      v[1] = 255;
      v[2] = 0;
    }
  }
}

void draw_lines(Mat &img, cv::Vec3f (&lines)[TOP_LINES * 2]) {
  float a;
  char p;
  hls::Scalar<unsigned char, 3> pixelh;       //pixel to add to hough image
  hls::Scalar<unsigned char, 3> pixela;       //pixel to read image
  int rows = img.rows;
  int cols = img.cols;

  for (int y = 0; y < rows; y++) {
    for (int x = 0; x < cols; x++) {
      p = 0;
      Vec<uchar, 3> &v = img.at<Vec<uchar, 3> >(y, x);
      for (int l=0; l<TOP_LINES*2; l++) {
        a = lines[l][0]*x +lines[l][1]*y - lines[l][2];
        if (-1 < a and a < 1 and (lines[l][0] != 0 or lines[l][1] != 0 or lines[l][2] != 0))
          p = (char) (lines[l][1] < 0 ? 1 : 2);
      }
      if (p == 1) {
        v[0] = 255;
        v[1] = 0;
        v[2] = 0;
      }
      else if (p == 2) {
        v[0] = 0;
        v[1] = 0;
        v[2] = 255;
      }
    }
  }
}

Vec2f intersection(Vec3f l1, Vec3f l2) {
  // l1 = (cos1, sin1, rho1), l2 = (cos2, sin2, rho2)
  float D = l1[0] * l2[1] - l2[0] * l1[1];
  float Dx = l1[2] * l2[1] - l2[2] * l1[1];
  float Dy = l1[0] * l2[2] - l2[0] * l1[2];
  if (D != 0)
    return Vec2f(Dx / D, Dy / D);
  return Vec2f(0, PARALLEL_Y);
}

void extract_vanishing(
  stream_t &stream_in,
  stream_t &stream_out,
  Params *signature,
  Output *output) {

  int row_mid = signature->mask_mid_y;
  float mid_val = signature->mask_mid_value / 256.f;
  if (not mask_initialized) {
    for (int row = 0; row < mask.rows; row++) {
      float v = (row < row_mid) ? mid_val * row / row_mid : 0.2f +
                                                            (1 - mid_val) * (row - row_mid) / (mask.rows - row_mid);
      for (int col = 0; col < mask.cols; col++) {
        Vec<float, 1> &val = mask.at<Vec<float, 1> >(row, col);
        val[0] = v;
      }
    }
    mask_initialized = true;
  }

  // stream_in = stream_in.rowRange(signature->top_crop, stream_in.rows);

  // hls::AXIvideo2Mat(stream_in, img0)
  rgb_img_t src = stream_in;
  rgb_img_t gray;
  rgb_img_t edges;
  rgb_img_t blur;
  cv::cvtColor(src, gray, COLOR_BGR2GRAY);

  int width = stream_in.cols, height = stream_in.rows;

  Vec2f vanishingMin(stream_in.cols * 0.35f, stream_in.rows * 0.0f);
  Vec2f vanishingMax(stream_in.cols * 0.65f, stream_in.rows * 0.4f);

  /* ###########################
   * #    PROCESS THE IMAGE    #
   * ########################### */
  // To pass it in grayscale
  // remove noise,
  // apply edge detecting kernels
  // fade out the sky and other useless features at the top
  // and apply Hough line matching filter
  cv::GaussianBlur(gray, blur, Size(signature->blur_size, signature->blur_size), 0, 0);
  rgb_img_t after_threshold;
  cv::Sobel(blur, edges, -1, 1, 0, 3);
  cv::multiply(edges, mask, edges, 1, CV_8UC1);
  cv::threshold(edges, after_threshold, signature->edge_threshold, 0, THRESH_TOZERO);

  double polar_theta_limit = signature->theta_limit * M_PI / 180;
  vector<Vec2f> polar_lines;//(HOUGH_LINES_COUNT);
  //Vec2f polar_lines[HOUGH_LINES_COUNT];
  cv::HoughLines(after_threshold, polar_lines, 1., M_PI / 180, signature->hough_treshold, 0, 0, -polar_theta_limit,
                 +polar_theta_limit);


  /* #####################################
   * #    EXTRACT MOST RELEVANT LINES    #
   * ##################################### */
  // Once we have our lines, cache cos and sin values
  // and remove the lines that do not start from either the left or the right of the screen
  // depending on the way they are tilded
  vector<Vec3f> lines(HOUGH_LINES_COUNT);
  int correct_line_count = 0;
  Vec3f bot_line(0, 1, stream_in.rows);

  for (size_t i = 0; i < MIN(HOUGH_LINES_COUNT, polar_lines.size()); ++i) {
    // hls::cordic::sin_cos_range_redux_cordic(lines[i].angle, s, c);
    Vec3f line = Vec3f(cos(polar_lines[i][1]), sin(polar_lines[i][1]), polar_lines[i][0]);
    float bot_x = intersection(line, bot_line)[0];

    // Must be tilted to the right and starts from the left part of the screen
    // or tilted to the left and starts from the right part of the screen
    if ((polar_lines[i][1] > 0 and bot_x < width / 2) or (polar_lines[i][1] < 0 and bot_x > width / 2)) {
      lines[correct_line_count] = line;
      correct_line_count++;
    }
  }

  bool keep[HOUGH_LINES_COUNT] = {false};
  int keep_left = 0, keep_right = 0;

  bool run = true;
  for (size_t i = 0; i < correct_line_count and run; ++i) {
    bool i_is_left = lines[i][1] < 0; // sin < 0 <=> theta < 0

    for (size_t j = 0; j < correct_line_count and run; j++) {
      // only intersect with lines tilded the other way
      if (i_is_left != (lines[j][1] > 0))
        continue;

      Vec2f res = intersection(lines[i], lines[j]);

      if (
        vanishingMin[0] < res[0] and
        res[0] < vanishingMax[0] and
        vanishingMin[1] < res[1] and
        res[1] < vanishingMax[1]) {
        if (not keep[i]) {
          if (i_is_left) {
            if (keep_left < TOP_LINES) {
              keep[i] = true;
              keep_left++;
            }
          } else {
            if (keep_right < TOP_LINES) {
              keep[i] = true;
              keep_right++;
            }
          }
        }
        if (not keep[j]) {
          if (i_is_left) {
            if (keep_right < TOP_LINES) {
              keep[j] = true;
              keep_right++;
            }
          } else {
            if (keep_left < TOP_LINES) {
              keep[j] = true;
              keep_left++;
            }
          }
        }
        run = keep_left < TOP_LINES or keep_right < TOP_LINES;
      }
    }
  }
  Vec3f selected_lines[TOP_LINES*2];
  int selected_lines_index = 0;

  /* #####################################
   * #      COMPUTE VANISHING POINT      #
   * ##################################### */

  // Using the average intersection of the best left and right lines
  Vec2f vanishing_point;
  int acc_count = 0;
  for (size_t i = 0; i < correct_line_count; ++i) {
    if (not keep[i])
      continue;
    selected_lines[selected_lines_index] = lines[i];
    selected_lines_index ++;

    bool i_is_left = lines[i][1] < 0; // sin < 0 <=> theta < 0

    for (size_t j = i + 1; j < correct_line_count; ++j) {
      if (not keep[j] or (lines[j][1] > 0) != i_is_left)
        continue;

      Vec2f res = intersection(lines[i], lines[j]);

      if (res[1] <= PARALLEL_Y)
        continue;

      vanishing_point += res;
      acc_count++;
    }
  }
  vanishing_point /= acc_count;


  /* #####################################
   * # DISPLAY LINES AND VANISHING POINT #
   * ##################################### */

  rgb_img_t out;
  if (signature->display_raw)
    src.copyTo(out);
  else {
    cv::cvtColor(after_threshold, out, COLOR_GRAY2BGR);
  }
  if (signature->display_extra) {
    // //rgb_img_t out;
    // for (size_t i = 0; i < correct_line_count; ++i) {
    //   if (not keep[i])
    //     continue;
    //   double a = lines[i][0];
    //   double b = lines[i][1];
    //   double x0 = a * lines[i][2];
    //   double y0 = b * lines[i][2];
    //   line(out,
    //        Point(int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))),
    //        Point(int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))),
    //        b < 0 ? Scalar(0, 0, 255) : Scalar(255, 0, 0), 1, LINE_AA);
    // }
    // // Display the vanishing point
    // // circle(out, Point(vanishing_point), 1, Scalar(0, 255, 0), 10);

    draw_lines(out, selected_lines);
    draw_dot(out, Point(vanishing_point), 5);
  }
  signature->mean_x = vanishing_point[0];
  signature->mean_y = vanishing_point[1];

  // hls::Mat2AXIvideo(img3, stream_out);
  out.copyTo(stream_out);
}
