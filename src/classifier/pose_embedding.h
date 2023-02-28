/*
 * Copyright 2020 Google LLC. All rights reserved.
 *
 * Copyright 2023 NXP
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * ************************************************************************
 * This source code is based in the classification solution provided by the
 * MLKit available at:  https://github.com/googlesamples/mlkit/tree/master/
 *                      android/vision-quickstart/app/src/main/java/com/
 *                      google/mlkit/vision/demo/java/posedetector/
 *                      classification
 * ************************************************************************
 *
 */

#pragma once

#include <cstring>
#include <iostream>
#include <vector>
// #include <cmath>
#include <algorithm>

#include "../utils/pose_landmark.h"

class FullBodyPoseEmbedder {
  const size_t number_raw_points;
  const size_t number_keypoints;

  float torso_size_multiplier;
  Landmark landmark;

  void normalize_pose_landmarks();

  Keypoint get_pose_center();
  float get_pose_size();
  std::vector<Keypoint> get_pose_distance_embedding();

public:
  FullBodyPoseEmbedder(float torso_size_multiplier = 2.5);
  std::vector<Keypoint> get_embedding(const Landmark &landmark);
};
