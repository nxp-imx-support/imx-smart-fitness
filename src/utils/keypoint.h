/*
 * Copyright 2023 NXP
 * SPDX-License-Identifier: Apache-2.0
 *
 * Class for 3D Keypoint used for landmark in pose estimation model
 *
 */

#pragma once

#include <cmath>
#include <string>

class Keypoint {
  float x;
  float y;
  float z;

public:
  // Constructors
  Keypoint();
  Keypoint(const float &x, const float &y);
  Keypoint(const float &x, const float &y, const float &z);

  // Copy constructor
  Keypoint(const Keypoint &kp);

  // Getter for x, y and z
  float operator[](const std::string &key) const;

  // Distance operator 2D
  float operator^(const Keypoint &kp);

  // Assignment operator
  Keypoint &operator=(const Keypoint &kp);

  Keypoint operator+(const float &value);
  Keypoint operator-(const float &value);
  Keypoint operator*(const float &value);
  Keypoint operator/(const float &value);

  Keypoint operator+=(const float &value);
  Keypoint operator-=(const float &value);
  Keypoint operator*=(const float &value);
  Keypoint operator/=(const float &value);

  Keypoint operator+(const Keypoint &kp);
  Keypoint operator-(const Keypoint &kp);
  Keypoint operator*(const Keypoint &kp);
  Keypoint operator/(const Keypoint &kp);

  Keypoint operator+=(const Keypoint &kp);
  Keypoint operator-=(const Keypoint &kp);
  Keypoint operator*=(const Keypoint &kp);
  Keypoint operator/=(const Keypoint &kp);
};
