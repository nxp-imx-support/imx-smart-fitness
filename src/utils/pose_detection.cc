/*
 * Copyright 2023 NXP
 * SPDX-License-Identifier: Apache-2.0
 *
 * Class for pose detection from MediaPipe model
 *
 * Bounding box in each pose detection is currently set to the bounding box of
 * the detected face. However, 4 additional key points are available in each
 * detection, which are used to further calculate a (rotated) bounding box that
 * encloses the body region of interest. Among the 4 key points, the first two
 * are for identifying the full-body region, and the second two for upper body
 * only:
 *      Key point 0 - mid hip center
 *      Key point 1 - point that encodes size & rotation (for full body)
 *      Key point 2 - mid shoulder center
 *      Key point 3 - point that encodes size & rotation (for upper body)
 *
 */

#include "pose_detection.h"

// Default constructor
PoseDetection::PoseDetection()
    : score{0.0}, bbox(), mid_hip_center(), full_body_size_rotation() {}

void PoseDetection::set_score(const float &score) { this->score = score; }

void PoseDetection::set_bbox(const BoundingBox &bbox) { this->bbox = bbox; }

void PoseDetection::set_mid_hip_center(const Keypoint &kp) {
  mid_hip_center = kp;
}

void PoseDetection::set_full_body_size_rotation(const Keypoint &kp) {
  full_body_size_rotation = kp;
}

BoundingBox PoseDetection::get_bbox() { return this->bbox; }

float PoseDetection::get_score() { return this->score; }

Keypoint PoseDetection::get_mid_hip_center() { return this->mid_hip_center; }

Keypoint PoseDetection::get_full_body_size_rotation() {
  return this->full_body_size_rotation;
}
