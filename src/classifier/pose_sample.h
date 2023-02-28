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

#include <iostream>
#include <string>
#include <vector>

#include "pose_embedding.h"

class PoseSample {
  FullBodyPoseEmbedder pose_embedding;

  std::string name;
  std::string class_name;
  std::vector<Keypoint> embedding;

public:
  PoseSample();
  PoseSample(std::string name, std::string class_name,
             const Landmark &landmark);
  PoseSample(const PoseSample &pose_sample);

  PoseSample &operator=(const PoseSample &pose);

  std::string get_name();
  std::string get_class_name();
  std::vector<Keypoint> get_embedding();
};
