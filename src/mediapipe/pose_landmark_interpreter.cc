/*
 * Copyright 2023 NXP
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pose_landmark_interpreter.h"

PoseLandmarkInterpreter::PoseLandmarkInterpreter(const int &num_detections,
                                                 const int &num_keypoints)
    : score{0.0}, raw_landmarks{nullptr}, scale{256.0}, score_threshold{0.7},
      pose_landmark{} {
  this->num_detections = num_detections;
  this->num_keypoints = num_keypoints;

  raw_landmarks = new float[this->num_detections * this->num_keypoints];

  // Check if memory was allocated
  if (nullptr == raw_landmarks) {
    std::cerr << "Could not allocate memory to pose landmark's interpreter!\n";
    exit(-1);
  }

  // Initialize to zero
  memset(raw_landmarks, 0.0,
         sizeof(float) * this->num_detections * this->num_keypoints);
}

PoseLandmarkInterpreter::~PoseLandmarkInterpreter() {
  if (nullptr != raw_landmarks) {
    delete[] raw_landmarks;
    raw_landmarks = nullptr;
  }
}

void PoseLandmarkInterpreter::decode_predictions(const float *raw_landmarks,
                                                 float &score) {
  if (nullptr != raw_landmarks) {
    memcpy(this->raw_landmarks, raw_landmarks,
           sizeof(float) * num_detections * num_keypoints);

    // Apply sigmoid to score
    score = 1.0 / (1.0 + std::exp(-score));

    if (score > score_threshold) {
      decode_landmark();
    }
  }
}

void PoseLandmarkInterpreter::decode_landmark() {
  for (size_t i{0}; i < 33; i++) {
    Keypoint keypoint(raw_landmarks[i * num_keypoints + 0],
                      raw_landmarks[i * num_keypoints + 1],
                      raw_landmarks[i * num_keypoints + 2]);
    pose_landmark[i] = keypoint / scale;
  }
}

Landmark PoseLandmarkInterpreter::get_pose_landmark() { return pose_landmark; }
