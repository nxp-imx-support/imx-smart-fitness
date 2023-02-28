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

#include "classification_result.h"

ClassificationResult::ClassificationResult() : class_confidences{} {
  // class_confidences[]
}

float ClassificationResult::get_class_confidence(
    const std::string &class_name) {
  return (class_confidences.find(class_name) == class_confidences.end())
             ? 0
             : class_confidences.at(class_name);
}

std::string ClassificationResult::get_max_confidence_class() {
  std::map<std::string, float>::iterator iterator_map;
  float max_confidence = 0;
  std::string key_class = "";
  for (iterator_map = class_confidences.begin();
       iterator_map != class_confidences.end(); iterator_map++) {
    // "first" tiene la clave. "second" el valor
    std::string key = iterator_map->first;
    int value = iterator_map->second;
    if (value > max_confidence) {
      max_confidence = value;
      key_class = key;
    }
  }
  return key_class;
}

void ClassificationResult::increment_class_confidence(
    const std::string &class_name) {
  class_confidences[class_name] =
      (class_confidences.find(class_name) == class_confidences.end())
          ? 1
          : class_confidences.at(class_name) + 1;
}

void ClassificationResult::put_class_confidence(const std::string &class_name,
                                                const float &confidence) {
  class_confidences[class_name] = confidence;
}

std::vector<std::string> ClassificationResult::getKeys() {
  std::map<std::string, float>::iterator iterator_map;
  std::vector<std::string> keys;
  for (iterator_map = class_confidences.begin();
       iterator_map != class_confidences.end(); iterator_map++) {
    keys.push_back(iterator_map->first);
  }
  return keys;
}

bool ClassificationResult::has_key(const std::string &key) {
  if (class_confidences.find(key) == class_confidences.end()) {
    return false;
  }
  return true;
}
