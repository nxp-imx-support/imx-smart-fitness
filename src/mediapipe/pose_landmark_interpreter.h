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

#include "../utils/pose_landmark.h"

class PoseLandmarkInterpreter {
  float score;
  float *raw_landmarks;

  int num_detections;
  int num_keypoints;
  float scale;

  const float score_threshold;

  Landmark pose_landmark;

  void decode_landmark();

public:
  PoseLandmarkInterpreter(const int &num_detections = 195,
                          const int &num_keypoints = 5);
  ~PoseLandmarkInterpreter();

  void decode_predictions(const float *raw_landmarks, float &score);
  Landmark get_pose_landmark();
};
