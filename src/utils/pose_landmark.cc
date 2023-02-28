/*
 * Copyright 2023 NXP
 * SPDX-License-Identifier: Apache-2.0
 *
 * Class for 3D Keypoint landmark
 *
 */

#include "pose_landmark.h"

// Constructor
Landmark::Landmark()
    : nose{}, left_eye_inner{}, left_eye{}, left_eye_outer{}, right_eye_inner{},
      right_eye{}, right_eye_outer{}, left_ear{}, right_ear{}, left_mouth{},
      right_mouth{}, left_shoulder{}, right_shoulder{}, left_elbow{},
      right_elbow{}, left_wrist{}, right_wrist{}, left_pinky{}, right_pinky{},
      left_index{}, right_index{}, left_thumb{}, right_thumb{}, left_hip{},
      right_hip{}, left_knee{}, right_knee{}, left_ankle{},
      right_ankle{}, left_heel{}, right_heel{}, left_foot{}, right_foot{} {}

// Assignment operator
Landmark &Landmark::operator=(const Landmark &landmark) {
  if (this == &landmark)
    return *this;

  this->nose = landmark.nose;
  this->left_eye_inner = landmark.left_eye_inner;
  this->left_eye = landmark.left_eye;
  this->left_eye_outer = landmark.left_eye_outer;
  this->right_eye_inner = landmark.right_eye_inner;
  this->right_eye = landmark.right_eye;
  this->right_eye_outer = landmark.right_eye_outer;
  this->left_ear = landmark.left_ear;
  this->right_ear = landmark.right_ear;
  this->left_mouth = landmark.left_mouth;
  this->right_mouth = landmark.right_mouth;
  this->left_shoulder = landmark.left_shoulder;
  this->right_shoulder = landmark.right_shoulder;
  this->left_elbow = landmark.left_elbow;
  this->right_elbow = landmark.right_elbow;
  this->left_wrist = landmark.left_wrist;
  this->right_wrist = landmark.right_wrist;
  this->left_pinky = landmark.left_pinky;
  this->right_pinky = landmark.right_pinky;
  this->left_index = landmark.left_index;
  this->right_index = landmark.right_index;
  this->left_thumb = landmark.left_thumb;
  this->right_thumb = landmark.right_thumb;
  this->left_hip = landmark.left_hip;
  this->right_hip = landmark.right_hip;
  this->left_knee = landmark.left_knee;
  this->right_knee = landmark.right_knee;
  this->left_ankle = landmark.left_ankle;
  this->right_ankle = landmark.right_ankle;
  this->left_heel = landmark.left_heel;
  this->right_heel = landmark.right_heel;
  this->left_foot = landmark.left_foot;
  this->right_foot = landmark.right_foot;

  return *this;
}

// Assignment operator
Keypoint &Landmark::operator[](const int &index) {
  switch (index) {
  case 0:
    return nose;
  case 1:
    return left_eye_inner;
  case 2:
    return left_eye;
  case 3:
    return left_eye_outer;
  case 4:
    return right_eye_inner;
  case 5:
    return right_eye;
  case 6:
    return right_eye_outer;
  case 7:
    return left_ear;
  case 8:
    return right_ear;
  case 9:
    return left_mouth;
  case 10:
    return right_mouth;
  case 11:
    return left_shoulder;
  case 12:
    return right_shoulder;
  case 13:
    return left_elbow;
  case 14:
    return right_elbow;
  case 15:
    return left_wrist;
  case 16:
    return right_wrist;
  case 17:
    return left_pinky;
  case 18:
    return right_pinky;
  case 19:
    return left_index;
  case 20:
    return right_index;
  case 21:
    return left_thumb;
  case 22:
    return right_thumb;
  case 23:
    return left_hip;
  case 24:
    return right_hip;
  case 25:
    return left_knee;
  case 26:
    return right_knee;
  case 27:
    return left_ankle;
  case 28:
    return right_ankle;
  case 29:
    return left_heel;
  case 30:
    return right_heel;
  case 31:
    return left_foot;
  case 32:
    return right_foot;
  }
}

// Getter
Keypoint Landmark::operator()(const int &index) const {
  switch (index) {
  case 0:
    return nose;
  case 1:
    return left_eye_inner;
  case 2:
    return left_eye;
  case 3:
    return left_eye_outer;
  case 4:
    return right_eye_inner;
  case 5:
    return right_eye;
  case 6:
    return right_eye_outer;
  case 7:
    return left_ear;
  case 8:
    return right_ear;
  case 9:
    return left_mouth;
  case 10:
    return right_mouth;
  case 11:
    return left_shoulder;
  case 12:
    return right_shoulder;
  case 13:
    return left_elbow;
  case 14:
    return right_elbow;
  case 15:
    return left_wrist;
  case 16:
    return right_wrist;
  case 17:
    return left_pinky;
  case 18:
    return right_pinky;
  case 19:
    return left_index;
  case 20:
    return right_index;
  case 21:
    return left_thumb;
  case 22:
    return right_thumb;
  case 23:
    return left_hip;
  case 24:
    return right_hip;
  case 25:
    return left_knee;
  case 26:
    return right_knee;
  case 27:
    return left_ankle;
  case 28:
    return right_ankle;
  case 29:
    return left_heel;
  case 30:
    return right_heel;
  case 31:
    return left_foot;
  case 32:
    return right_foot;
  }
}

// Getter
Keypoint Landmark::operator[](const std::string &key) {
  if (key == "nose")
    return nose;
  if (key == "left_eye_inner")
    return left_eye_inner;
  if (key == "left_eye")
    return left_eye;
  if (key == "left_eye_outer")
    return left_eye_outer;
  if (key == "right_eye_inner")
    return right_eye_inner;
  if (key == "right_eye")
    return right_eye;
  if (key == "right_eye_outer")
    return right_eye_outer;
  if (key == "left_ear")
    return left_ear;
  if (key == "right_ear")
    return right_ear;
  if (key == "left_mouth")
    return left_mouth;
  if (key == "right_mouth")
    return right_mouth;
  if (key == "left_shoulder")
    return left_shoulder;
  if (key == "right_shoulder")
    return right_shoulder;
  if (key == "left_elbow")
    return left_elbow;
  if (key == "right_elbow")
    return right_elbow;
  if (key == "left_wrist")
    return left_wrist;
  if (key == "right_wrist")
    return right_wrist;
  if (key == "left_pinky")
    return left_pinky;
  if (key == "right_pinky")
    return right_pinky;
  if (key == "left_index")
    return left_index;
  if (key == "right_index")
    return right_index;
  if (key == "left_thumb")
    return left_thumb;
  if (key == "right_thumb")
    return right_thumb;
  if (key == "left_hip")
    return left_hip;
  if (key == "right_hip")
    return right_hip;
  if (key == "left_knee")
    return left_knee;
  if (key == "right_knee")
    return right_knee;
  if (key == "left_ankle")
    return left_ankle;
  if (key == "right_ankle")
    return right_ankle;
  if (key == "left_heel")
    return left_heel;
  if (key == "right_heel")
    return right_heel;
  if (key == "left_foot")
    return left_foot;
  if (key == "right_foot")
    return right_foot;
}

Landmark Landmark::operator*(const float &factor) {
  Landmark tmp_lm(*this);

  for (size_t i{0}; i < 33; i++)
    tmp_lm[i] = tmp_lm(i) * factor;

  return tmp_lm;
}

Landmark Landmark::operator/(const Landmark &landmark) {
  Landmark tmp_lm(*this);

  for (size_t i{0}; i < 33; i++)
    tmp_lm[i] = tmp_lm(i) / landmark(i);

  return tmp_lm;
}

Landmark Landmark::operator+=(const Landmark &landmark) {
  for (size_t i{0}; i < 33; i++)
    (*this)[i] = (*this)(i) + landmark(i);

  return *this;
}

Landmark Landmark::operator+=(const float &factor) {
  for (size_t i{0}; i < 33; i++)
    (*this)[i] = (*this)(i) + factor;

  return *this;
}
