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

#include "pose_sample.h"

PoseSample::PoseSample() : pose_embedding() {
  this->name = "";
  this->class_name = "";
}

PoseSample::PoseSample(std::string name, std::string class_name,
                       const Landmark &landmark)
    : PoseSample() {
  this->name = name;
  this->class_name = class_name;
  this->embedding = pose_embedding.get_embedding(landmark);
}

PoseSample::PoseSample(const PoseSample &pose_sample) : PoseSample() {
  this->name = pose_sample.name;
  this->class_name = pose_sample.class_name;
  this->embedding = pose_sample.embedding;
}

PoseSample &PoseSample::operator=(const PoseSample &pose) {
  if (this == &pose) {
    return *this;
  }

  this->name = pose.name;
  this->class_name = pose.class_name;
  this->embedding = pose.embedding;

  return *this;
}

std::string PoseSample::get_name() { return this->name; }
std::string PoseSample::get_class_name() { return this->class_name; }
std::vector<Keypoint> PoseSample::get_embedding() { return this->embedding; }
