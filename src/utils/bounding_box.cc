/*
 * Copyright 2023 NXP
 * SPDX-License-Identifier: Apache-2.0
 *
 * Class for Bounding Box object
 *
 * BOUNDING BOX IS DEFINED AS BELOW:
 *
 *    (xmin, ymin) *----------
 *                 |          |
 *                 |          |
 *                 |          |
 *                 |          |
 *                 ----------* (xmax, ymax)
 *
 */

#include "bounding_box.h"

/*
 * Default constructor
 */
BoundingBox::BoundingBox() : xmin{0.0}, ymin{0.0}, xmax{0.0}, ymax{0.0} {}

/**
 * Constructor for separate values
 */
BoundingBox::BoundingBox(const float &xmin, const float &ymin,
                         const float &xmax, const float &ymax)
    : BoundingBox() {
  this->xmin = xmin;
  this->ymin = ymin;
  this->xmax = xmax;
  this->ymax = ymax;
}

/**
 * Constructor with keypoints
 */
BoundingBox::BoundingBox(const Keypoint &min_kp, const Keypoint &max_kp) {
  this->xmin = min_kp["x"];
  this->ymin = min_kp["y"];
  this->xmax = max_kp["x"];
  this->ymax = max_kp["y"];
}

/**
 * Copy Constructor
 */
BoundingBox::BoundingBox(const BoundingBox &bbox) : BoundingBox() {
  this->xmin = bbox.xmin;
  this->ymin = bbox.ymin;
  this->xmax = bbox.xmax;
  this->ymax = bbox.ymax;
}

/**
 * Getters for points
 */
float BoundingBox::operator()(const std::string &key) const {
  if (key == "xmin")
    return this->xmin;
  else if (key == "ymin")
    return this->ymin;
  else if (key == "xmax")
    return this->xmax;
  else if (key == "ymax")
    return this->ymax;
  else
    throw std::string{"\tIndex out of range!"};
}

/**
 * Set operator [] for assignment
 */
float &BoundingBox::operator[](const std::string &key) {
  if (key == "xmin")
    return this->xmin;
  else if (key == "ymin")
    return this->ymin;
  else if (key == "xmax")
    return this->xmax;
  else if (key == "ymax")
    return this->ymax;
  else
    throw std::string{"\tIndex out of range!"};
}

/**
 * Assignment operator
 */
BoundingBox &BoundingBox::operator=(const BoundingBox &bbox) {
  if (this == &bbox)
    return *this;

  this->xmin = bbox.xmin;
  this->ymin = bbox.ymin;
  this->xmax = bbox.xmax;
  this->ymax = bbox.ymax;

  return *this;
}

/**
 * Offset operator
 */
BoundingBox BoundingBox::operator+=(const float &offset) {
  this->xmin += offset;
  this->ymin += offset;
  this->xmax += offset;
  this->ymax += offset;

  return *this;
}

/**
 * Sum operator
 */
BoundingBox BoundingBox::operator+=(const BoundingBox &bbox) {
  this->xmin += bbox.xmin;
  this->ymin += bbox.ymin;
  this->xmax += bbox.xmax;
  this->ymax += bbox.ymax;
  return *this;
}

/**
 * Multiply operator
 */
BoundingBox BoundingBox::operator*(const float &offset) {
  BoundingBox bbox(this->xmin * offset, this->ymin * offset,
                   this->xmax * offset, this->ymax * offset);
  return bbox;
}

/**
 * Divide operator
 */
BoundingBox BoundingBox::operator/(const BoundingBox &bbox) {
  BoundingBox tmp_bbox(this->xmin / bbox.xmin, this->ymin / bbox.ymin,
                       this->xmax / bbox.xmax, this->ymax / bbox.ymax);
  return tmp_bbox;
}
