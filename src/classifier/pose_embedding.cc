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

#include "pose_embedding.h"

FullBodyPoseEmbedder::FullBodyPoseEmbedder(float torso_size_multiplier)
    : number_raw_points{99}, number_keypoints{33}, landmark{} {
  this->torso_size_multiplier = torso_size_multiplier;
}

/*
   Normalizes pose landmarks and converts to embedding

Args:
landmarks - 3D landmarks of shape (N, 3).

Result:
Vector with pose embedding of shape (M, 3) where `M` is the number of
pairwise distances.
*/
std::vector<Keypoint>
FullBodyPoseEmbedder::get_embedding(const Landmark &landmark) {
  // Copy landmarks to new array
  this->landmark = landmark;

  // Normalize pose landmarks
  normalize_pose_landmarks();
  return get_pose_distance_embedding();
}

void FullBodyPoseEmbedder::normalize_pose_landmarks() {
  // Normalize translation and scale
  Keypoint pose_center((landmark["left_hip"] + landmark["right_hip"]) * 0.5);
  float pose_size = get_pose_size();

  for (size_t i{0}; i < number_keypoints; i++)
    landmark[i] = ((landmark(i) - pose_center) / pose_size) * 100;
}

float FullBodyPoseEmbedder::get_pose_size() {
  // Calculates pose size.
  //  It is the maximum of two values:
  //       * Torso size multiplied by `torso_size_multiplier`
  //       * Maximum distance from pose center to any pose landmark

  Keypoint hips_center((landmark["left_hip"] + landmark["right_hip"]) * 0.5);
  Keypoint shoulders_center(
      (landmark["left_shoulder"] + landmark["right_shoulder"]) * 0.5);

  // Torso size as the minimum body size (L2 norm on 2D, z is not used)
  float torso_size = hips_center ^ shoulders_center;

  // Max dist to pose center (L2 norm on 2D, z is not used)
  float max_distance = torso_size * torso_size_multiplier;
  for (size_t i{0}; i < number_keypoints; i++) {
    float distance = landmark(i) ^ hips_center;
    if (distance > max_distance)
      max_distance = distance;
  }
  return max_distance;
}

std::vector<Keypoint> FullBodyPoseEmbedder::get_pose_distance_embedding() {
  // Converts pose landmarks into 3D embedding.
  //
  // We use several pairwise 3D distances to form pose embedding. All distances
  // include X and Y components with sign. We differnt types of pairs to cover
  // different pose classes. Feel free to remove some or add new.
  //
  // Args:
  // landmarks - NumPy array with 3D landmarks of shape (N, 3).
  //
  // Result:
  // Numpy array with pose embedding of shape (M, 3) where `M` is the number of
  // pairwise distances.
  std::vector<Keypoint> embedding;

  // **ONE JOINT

  // Get average distance from left_hip to right_hip and left_shoulder to
  // right_shoulder
  Keypoint average_hip((landmark["left_hip"] + landmark["right_hip"]) * 0.5);
  Keypoint average_shoulder(
      (landmark["left_shoulder"] + landmark["right_shoulder"]) * 0.5);
  embedding.push_back(average_shoulder - average_hip);

  // Get distance from left_shoulder to left_elbow
  embedding.push_back((landmark["left_shoulder"] + landmark["left_elbow"]) *
                      0.5);
  // Get distance from right_shoulder to right_elbow
  embedding.push_back((landmark["right_shoulder"] + landmark["right_elbow"]) *
                      0.5);

  // Get distance from left_elbow to left_wrist
  embedding.push_back((landmark["left_elbow"] + landmark["left_wrist"]) * 0.5);
  // Get distance from right_elbow to right_wrist
  embedding.push_back((landmark["right_elbow"] + landmark["right_wrist"]) *
                      0.5);

  // Get distance from left_hip to left_knee
  embedding.push_back((landmark["left_hip"] + landmark["left_knee"]) * 0.5);
  // Get distance from right_hip to right_knee
  embedding.push_back((landmark["right_hip"] + landmark["right_knee"]) * 0.5);

  // Get distance from left_knee to left_ankle
  embedding.push_back((landmark["left_knee"] + landmark["left_ankle"]) * 0.5);
  // Get distance from right_knee to right_ankle
  embedding.push_back((landmark["right_knee"] + landmark["right_ankle"]) * 0.5);

  // **TWO JOINTS

  // Get distance from left_shoulders to left_wrist
  embedding.push_back((landmark["left_shoulders"] + landmark["left_wrist"]) *
                      0.5);
  // Get distance from right_shoulders to right_wrist
  embedding.push_back((landmark["right_shoulders"] + landmark["right_wrist"]) *
                      0.5);

  // Get distance from left_hip to left_ankle
  embedding.push_back((landmark["left_hip"] + landmark["left_ankle"]) * 0.5);
  // Get distance from right_hip to right_ankle
  embedding.push_back((landmark["right_hip"] + landmark["right_ankle"]) * 0.5);

  // **FOUR JOINTS

  // Get distance from left_hip to left_wrist
  embedding.push_back((landmark["left_hip"] + landmark["left_wrist"]) * 0.5);
  // Get distance from right_hip to right_wrist
  embedding.push_back((landmark["right_hip"] + landmark["right_wrist"]) * 0.5);

  // **FIVE JOINTS

  // Get distance from left_shoulders to left_ankle
  embedding.push_back((landmark["left_shoulders"] + landmark["left_ankle"]) *
                      0.5);
  // Get distance from right_shoulders to right_ankle
  embedding.push_back((landmark["right_shoulders"] + landmark["right_ankle"]) *
                      0.5);

  // Get distance from left_hip to left_wrist
  embedding.push_back((landmark["left_hip"] + landmark["left_wrist"]) * 0.5);
  // Get distance from right_hip to right_wrist
  embedding.push_back((landmark["right_hip"] + landmark["right_wrist"]) * 0.5);

  // ** CROSS BODY

  // Get distance from left_elbow to right_elbow
  embedding.push_back((landmark["left_elbow"] + landmark["right_elbow"]) * 0.5);
  // Get distance from left_knee to right_knee
  embedding.push_back((landmark["left_knee"] + landmark["right_knee"]) * 0.5);

  // Get distance from left_wrist to right_wrist
  embedding.push_back((landmark["left_wrist"] + landmark["right_wrist"]) * 0.5);
  // Get distance from left_ankle to right_ankle
  embedding.push_back((landmark["left_ankle"] + landmark["right_ankle"]) * 0.5);

  return embedding;
}
