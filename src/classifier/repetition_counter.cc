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

#include "repetition_counter.h"

RepetitionCounter::RepetitionCounter(const std::string &class_name,
                                     const int &enter_threshold,
                                     const int &exit_threshold) {
  this->class_name = class_name;
  this->enter_threshold = enter_threshold;
  this->exit_threshold = exit_threshold;
  pose_entered = false;
  n_repeats = 0;
}

int RepetitionCounter::count(ClassificationResult &result) {
  n_repeats = n_repeats % 12;

  // Get pose confidence
  float pose_confidence = 0.0;
  if (result.has_key(class_name)) {
    pose_confidence = result.get_class_confidence(class_name);
  }

  // On the very first frame or if we were out of the pose, just check if we
  // entered it on this frame and update the state.
  if (!pose_entered) {
    pose_entered = pose_confidence > enter_threshold;
    return n_repeats;
  }

  // If we were in the pose and are exiting it, then increase the counter and
  // update the state.
  if (pose_confidence < exit_threshold) {
    n_repeats += 1;
    pose_entered = false;
  }

  return n_repeats;
}
