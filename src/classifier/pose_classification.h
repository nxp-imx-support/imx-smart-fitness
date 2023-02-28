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

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "classification_result.h"
#include "classification_smoothing.h"
#include "pose_embedding.h"
#include "pose_sample.h"

class PoseClassifier {
  FullBodyPoseEmbedder pose_embedding;
  std::vector<PoseSample> pose_samples;
  const size_t top_n_by_max_distance;
  const size_t top_n_by_mean_distance;

  void load_pose_samples(const char *embeddings_file);
  float getMaxAbs(const Keypoint &point);
  float getSumAbs(const Keypoint &point);

public:
  PoseClassifier(const char *embeddings_file);

  ClassificationResult classify_pose(const Landmark &landmark);
};
