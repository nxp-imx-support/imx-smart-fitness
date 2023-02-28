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

#pragma once

#include "bounding_box.h"

class PoseDetection {
  float score;
  BoundingBox bbox;
  Keypoint mid_hip_center;
  Keypoint full_body_size_rotation;

public:
  // Constructor
  PoseDetection();

  // Setters
  void set_score(const float &score);
  void set_bbox(const BoundingBox &bbox);
  void set_mid_hip_center(const Keypoint &kp);
  void set_full_body_size_rotation(const Keypoint &kp);

  // Getters
  BoundingBox get_bbox();
  float get_score();
  Keypoint get_mid_hip_center();
  Keypoint get_full_body_size_rotation();
};
