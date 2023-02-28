/*
 * Copyright 2023 NXP
 * SPDX-License-Identifier: Apache-2.0
 *
 * Class for 3D Keypoint used for landmark in pose estimation model
 *
 */

#include "keypoint.h"

// Default constructor
Keypoint::Keypoint() : x{0.0}, y{0.0}, z{0.0} {}

// Constructor with values for x and y
Keypoint::Keypoint(const float &x, const float &y) : Keypoint() {
  this->x = x;
  this->y = y;
}

// Constructor with values for x, y and z
Keypoint::Keypoint(const float &x, const float &y, const float &z)
    : Keypoint() {
  this->x = x;
  this->y = y;
  this->z = z;
}

// Copy constructor
Keypoint::Keypoint(const Keypoint &kp) : Keypoint() {
  this->x = kp.x;
  this->y = kp.y;
  this->z = kp.z;
}

// Getter for x, y and z
float Keypoint::operator[](const std::string &key) const {
  if (key == "x") {
    return this->x;
  }
  if (key == "y") {
    return this->y;
  }
  if (key == "z") {
    return this->z;
  }

  // TODO: Throw exeption
  return 0.0;
}

// Distance operator 2D
float Keypoint::operator^(const Keypoint &kp) {
  return std::sqrt(std::pow(this->x - kp.x, 2) + std::pow(this->y - kp.y, 2));
}

// Assignment operator
Keypoint &Keypoint::operator=(const Keypoint &kp) {
  if (this == &kp) {
    return *this;
  }

  this->x = kp.x;
  this->y = kp.y;
  this->z = kp.z;

  return *this;
}

Keypoint Keypoint::operator+(const float &value) {
  Keypoint tmp_kp(this->x + value, this->y + value, this->z + value);
  return tmp_kp;
}

Keypoint Keypoint::operator-(const float &value) {
  Keypoint tmp_kp(this->x - value, this->y - value, this->z - value);
  return tmp_kp;
}

Keypoint Keypoint::operator*(const float &value) {
  Keypoint tmp_kp(this->x * value, this->y * value, this->z * value);
  return tmp_kp;
}

Keypoint Keypoint::operator/(const float &value) {
  Keypoint tmp_kp(this->x / value, this->y / value, this->z / value);
  return tmp_kp;
}

Keypoint Keypoint::operator+=(const float &value) {
  this->x += value;
  this->y += value;
  this->z += value;

  return *this;
}

Keypoint Keypoint::operator-=(const float &value) {
  this->x -= value;
  this->y -= value;
  this->z -= value;

  return *this;
}

Keypoint Keypoint::operator*=(const float &value) {
  this->x *= value;
  this->y *= value;
  this->z *= value;

  return *this;
}

Keypoint Keypoint::operator/=(const float &value) {
  this->x /= value;
  this->y /= value;
  this->z /= value;

  return *this;
}

Keypoint Keypoint::operator+(const Keypoint &kp) {
  Keypoint tmp_kp(this->x + kp.x, this->y + kp.y, this->z + kp.z);
  return tmp_kp;
}

Keypoint Keypoint::operator-(const Keypoint &kp) {
  Keypoint tmp_kp(this->x - kp.x, this->y - kp.y, this->z - kp.z);
  return tmp_kp;
}

Keypoint Keypoint::operator*(const Keypoint &kp) {
  Keypoint tmp_kp(this->x * kp.x, this->y * kp.y, this->z * kp.z);
  return tmp_kp;
}

Keypoint Keypoint::operator/(const Keypoint &kp) {
  Keypoint tmp_kp(this->x / kp.x, this->y / kp.y, this->z / kp.z);
  return tmp_kp;
}

Keypoint Keypoint::operator+=(const Keypoint &kp) {
  this->x += kp.x;
  this->y += kp.y;
  this->z += kp.z;

  return *this;
}

Keypoint Keypoint::operator-=(const Keypoint &kp) {
  this->x -= kp.x;
  this->y -= kp.y;
  this->z -= kp.z;

  return *this;
}

Keypoint Keypoint::operator*=(const Keypoint &kp) {
  this->x *= kp.x;
  this->y *= kp.y;
  this->z *= kp.z;

  return *this;
}

Keypoint Keypoint::operator/=(const Keypoint &kp) {
  this->x /= kp.x;
  this->y /= kp.y;
  this->z /= kp.z;

  return *this;
}
