/*
 * Copyright 2023 NXP
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include "bounding_box.h"
#include "pose_landmark.h"

class Filter {
  size_t window_size;
  float alpha;
  float alpha_landmarks;
  std::vector<BoundingBox> data;
  std::vector<Landmark> data_landmark;

public:
  Filter();
  BoundingBox filter(BoundingBox &detection);
  Landmark filter(Landmark &landmark);
};
