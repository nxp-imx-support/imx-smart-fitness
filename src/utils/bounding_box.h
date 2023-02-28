/*
 * Copyright 2023 NXP
 * SPDX-License-Identifier: Apache-2.0
 *
 * Class for Bounding Box object
 *
 * BOUNDING BOX IS DEFINED AS BELOW:
 *
 *    (xmin, ymin) *----------
 *                 |          |
 *                 |          |
 *                 |          |
 *                 |          |
 *                 ----------* (xmax, ymax)
 *
 */

#pragma once
#include "keypoint.h"

/**
 * Class definition for BoundingBox
 */
class BoundingBox {
  float xmin;
  float ymin;
  float xmax;
  float ymax;

public:
  BoundingBox();
  BoundingBox(const float &xmin, const float &ymin, const float &xmax,
              const float &ymax);
  BoundingBox(const Keypoint &min_kp, const Keypoint &max_kp);
  BoundingBox(const BoundingBox &bbox);

  // Getter for points
  float operator()(const std::string &key) const;

  // Set operator [] for assignment
  float &operator[](const std::string &key);

  // Assignment operator
  BoundingBox &operator=(const BoundingBox &bbox);

  // Overloading operators for bounding box computing
  BoundingBox operator+=(const float &offset);
  BoundingBox operator+=(const BoundingBox &bbox);
  BoundingBox operator*(const float &offset);
  BoundingBox operator/(const BoundingBox &bbox);
};
