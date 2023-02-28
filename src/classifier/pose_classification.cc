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

#include "pose_classification.h"

PoseClassifier::PoseClassifier(const char *embeddings_file)
    : pose_embedding{}, top_n_by_max_distance{30}, top_n_by_mean_distance{10} {
  load_pose_samples(embeddings_file);
}

void PoseClassifier::load_pose_samples(const char *embeddings_file) {
  std::fstream file_in;
  std::string line, word;
  std::vector<std::string> row;
  std::vector<std::vector<std::string>> content;

  // Open CSV file
  file_in.open(embeddings_file, std::ios::in);

  if (file_in.is_open()) {
    while (getline(file_in, line)) {
      row.clear();
      std::stringstream str(line);

      while (getline(str, word, ','))
        row.push_back(word);

      content.push_back(row);
    }
  } else
    std::cout << "Could not open the pose embeddings file\n";

  file_in.close();

  // Generate pose_samples
  Landmark landmark;
  pose_samples.clear();

  for (size_t i{0}; i < content.size(); i++) {
    for (size_t j{0}; j < 33; j++) {
      Keypoint kp(std::stof(content.at(i).at(2 + (j * 3 + 0))),
                  std::stof(content.at(i).at(2 + (j * 3 + 1))),
                  std::stof(content.at(i).at(2 + (j * 3 + 2))));
      landmark[j] = kp;
    }

    // Temporary normalization
    for (size_t z{0}; z < 33; z++) {
      Keypoint tmp_kp(landmark(z)["x"] / 1920.0, landmark(z)["y"] / 1080.0,
                      landmark(z)["z"] / 1920.0);
      landmark[z] = tmp_kp;
    }

    PoseSample sample(content.at(i).at(0), content.at(i).at(1), landmark);
    pose_samples.push_back(sample);
  }
}

ClassificationResult PoseClassifier::classify_pose(const Landmark &landmark) {
  // Create optimized funtion for flipping landmarks
  Landmark flipped_landmarks = landmark;
  for (size_t i{0}; i < 33; i++) {
    Keypoint kp(flipped_landmarks(i)["x"] * -1.0, flipped_landmarks(i)["y"],
                flipped_landmarks(i)["z"]);
    flipped_landmarks[i] = kp;
  }

  // Get pose embedding
  std::vector<Keypoint> embeddings = pose_embedding.get_embedding(landmark);
  std::vector<Keypoint> flipped_embeddings =
      pose_embedding.get_embedding(flipped_landmarks);

  // Filter by max distance
  //
  // That helps to remove outliers - poses that are almost the same as the
  // given one, but has one joint bent into another direction and actually
  // represnt a different pose class.

  std::vector<std::pair<PoseSample, float>> maxDistances;

  Keypoint scale{1.0, 1.0, 0.2};
  for (size_t i{0}; i < pose_samples.size(); i++) {
    float originalMax{0};
    float flippedMax{0};

    for (size_t j{0}; j < embeddings.size(); j++) {
      originalMax = std::max(
          originalMax, getMaxAbs((embeddings.at(j) -
                                  pose_samples.at(i).get_embedding().at(j)) *
                                 scale));
      flippedMax = std::max(
          flippedMax, getMaxAbs((flipped_embeddings.at(j) -
                                 pose_samples.at(i).get_embedding().at(j)) *
                                scale));
    }
    maxDistances.push_back(std::pair<PoseSample, float>(
        pose_samples.at(i), std::min(originalMax, flippedMax)));
  }

  // Sort maxDistances with max ontop
  std::sort(maxDistances.begin(), maxDistances.end(),
            [](auto &left, auto &right) { return left.second > right.second; });

  // Remove max distance
  if (maxDistances.size() > top_n_by_max_distance) {
    size_t num_to_remove = maxDistances.size() - top_n_by_max_distance;
    for (size_t i{0}; i < num_to_remove; i++)
      maxDistances.erase(maxDistances.begin());
  }

  // Filter by mean d istance.
  // After removing outliers we can find the nearest pose by mean distance.

  std::vector<std::pair<PoseSample, float>> meanDistances;

  for (size_t i{0}; i < maxDistances.size(); i++) {
    float originalSum{0};
    float flippedSum{0};

    PoseSample poseSample(maxDistances.at(i).first);
    for (size_t j{0}; j < embeddings.size(); j++) {
      originalSum += getSumAbs(
          (embeddings.at(j) - poseSample.get_embedding().at(j)) * scale);
      flippedSum += getSumAbs(
          (flipped_embeddings.at(j) - poseSample.get_embedding().at(j)) *
          scale);
    }
    meanDistances.push_back(std::pair<PoseSample, float>(
        maxDistances.at(i).first,
        std::min(originalSum, flippedSum) / (embeddings.size() * 2)));
  }

  // Sort maxDistances with max ontop
  std::sort(meanDistances.begin(), meanDistances.end(),
            [](auto &left, auto &right) { return left.second > right.second; });

  // Remove max distance
  if (meanDistances.size() > top_n_by_mean_distance) {
    size_t num_to_remove = meanDistances.size() - top_n_by_mean_distance;
    for (size_t i{0}; i < num_to_remove; i++)
      meanDistances.erase(meanDistances.begin());
  }

  ClassificationResult classification_result;
  for (size_t i{0}; i < meanDistances.size(); i++) {
    std::string class_name = meanDistances.at(i).first.get_class_name();
    classification_result.increment_class_confidence(class_name);
  }

  std::string class_output = classification_result.get_max_confidence_class();
  return classification_result;
}

float PoseClassifier::getMaxAbs(const Keypoint &point) {
  return std::max(
      {std::abs(point["x"]), std::abs(point["y"]), std::abs(point["z"])});
}

float PoseClassifier::getSumAbs(const Keypoint &point) {
  return std::abs(point["x"]) + std::abs(point["y"]) + std::abs(point["z"]);
}
