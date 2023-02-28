/*
 * Copyright 2023 NXP
 * SPDX-License-Identifier: Apache-2.0
 *
 * Class for 3D Keypoint landmark
 *
 */

#pragma once

#include "keypoint.h"

class Landmark {
  Keypoint nose;
  Keypoint left_eye_inner;
  Keypoint left_eye;
  Keypoint left_eye_outer;
  Keypoint right_eye_inner;
  Keypoint right_eye;
  Keypoint right_eye_outer;
  Keypoint left_ear;
  Keypoint right_ear;
  Keypoint left_mouth;
  Keypoint right_mouth;
  Keypoint left_shoulder;
  Keypoint right_shoulder;
  Keypoint left_elbow;
  Keypoint right_elbow;
  Keypoint left_wrist;
  Keypoint right_wrist;
  Keypoint left_pinky;
  Keypoint right_pinky;
  Keypoint left_index;
  Keypoint right_index;
  Keypoint left_thumb;
  Keypoint right_thumb;
  Keypoint left_hip;
  Keypoint right_hip;
  Keypoint left_knee;
  Keypoint right_knee;
  Keypoint left_ankle;
  Keypoint right_ankle;
  Keypoint left_heel;
  Keypoint right_heel;
  Keypoint left_foot;
  Keypoint right_foot;

public:
  Landmark();

  // Assignment operators
  Landmark &operator=(const Landmark &landmark);
  Keypoint &operator[](const int &index);

  // Getters
  Keypoint operator()(const int &index) const;
  Keypoint operator[](const std::string &key);

  Landmark operator*(const float &factor);
  Landmark operator/(const Landmark &landmark);
  Landmark operator+=(const Landmark &landmark);
  Landmark operator+=(const float &factor);
};
