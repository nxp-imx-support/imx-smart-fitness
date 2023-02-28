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

#include "classification_smoothing.h"

EMAFilter::EMAFilter() : window_size{10}, alpha{0.2}, data() {}

ClassificationResult EMAFilter::filter(ClassificationResult &detection) {
  data.insert(data.begin(), detection);
  if (data.size() > window_size) {
    data.pop_back();
  }

  std::vector<std::string> keys = detection.getKeys();

  ClassificationResult smoothed_data;
  for (size_t i{0}; i < keys.size(); i++) {
    float factor = 1.0;
    float top_sum = 0.0;
    float bottom_sum = 0.0;
    for (size_t j{0}; j < data.size(); j++) {
      float confidence = 0.0;
      if (data.at(j).has_key(keys.at(i)))
        confidence = (data.at(j)).get_class_confidence(keys.at(i));
      top_sum += factor * confidence;
      bottom_sum += factor;
      factor *= (1.0 - alpha);
    }
    smoothed_data.put_class_confidence(keys.at(i), top_sum / bottom_sum);
  }
  return smoothed_data;
}
