/*
 * Copyright 2023 NXP
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <string.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

#include "../utils/pose_detection.h"

class PoseDetectionInterpreter {
  float *scores;
  float *raw_bbox;

  size_t num_detections;
  size_t num_keypoints;
  float scale;

  const float score_threshold;
  const float nms_threshold;
  std::vector<float> anchors;

  std::vector<PoseDetection> detected_poses; // Detected poses (decoded result)

  // Apply sigmoid to scores
  void decode_scores();

  std::vector<float> load_anchors(char const *filename);

  BoundingBox decode_bbox(const size_t &index);
  Keypoint decode_mid_hip_center(const size_t &index);
  Keypoint decode_full_body_size_rotation(const size_t &index);

  std::vector<PoseDetection> nms(std::vector<PoseDetection> &beposesfore,
                                 const float &nms_threshold);
  float iou(const BoundingBox &rectA, const BoundingBox &rectB);
  static bool comparer(PoseDetection &score_a, PoseDetection &score_b);

public:
  PoseDetectionInterpreter(const char *anchors_file,
                           const int &num_detections = 2254,
                           const int &num_keypoints = 12);
  ~PoseDetectionInterpreter();

  void decode_predictions(const float *raw_bbox, const float *scores);
  std::vector<PoseDetection> get_pose_detections();
};
