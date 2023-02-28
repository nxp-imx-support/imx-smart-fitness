/*
 * Copyright 2023 NXP
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ema_filter.h"

Filter::Filter()
    : window_size{10}, alpha{0.1}, alpha_landmarks{0.4}, data(),
      data_landmark() {}

BoundingBox Filter::filter(BoundingBox &detection) {
  data.insert(data.begin(), detection);
  if (data.size() > window_size) {
    data.pop_back();
  }

  BoundingBox smoothed_box;
  float factor = 1.0;

  BoundingBox top_sum;
  BoundingBox bottom_sum;

  for (size_t i{0}; i < data.size(); i++) {
    top_sum += (data.at(i) * factor);
    bottom_sum += factor;
    factor *= (1.0 - alpha);
  }

  smoothed_box = top_sum / bottom_sum;

  return smoothed_box;
}

Landmark Filter::filter(Landmark &landmark) {
  data_landmark.insert(data_landmark.begin(), landmark);
  if (data_landmark.size() > window_size) {
    data_landmark.pop_back();
  }

  Landmark smoothed_box;
  float factor = 1.0;

  Landmark top_sum;
  Landmark bottom_sum;

  for (size_t i{0}; i < data_landmark.size(); i++) {
    top_sum += (data_landmark.at(i) * factor);
    bottom_sum += factor;
    factor *= (1.0 - alpha_landmarks);
  }

  smoothed_box = top_sum / bottom_sum;

  return smoothed_box;
}
