/*
 * Copyright 2023 NXP
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pose_detection_interpreter.h"

PoseDetectionInterpreter::PoseDetectionInterpreter(const char *anchors_file,
                                                   const int &num_detections,
                                                   const int &num_keypoints)
    : scores{nullptr}, raw_bbox{nullptr}, scale{224.0}, score_threshold{0.5},
      nms_threshold{0.3} {
  this->num_detections = num_detections;
  this->num_keypoints = num_keypoints;

  scores = new float[this->num_detections];
  raw_bbox = new float[this->num_detections * this->num_keypoints];

  // Check if memory was allocated
  if (nullptr == scores || nullptr == raw_bbox) {
    std::cerr << "Could not allocate memory to pose interpreter!\n";
    exit(-1);
  }

  // Initialize to zero
  memset(scores, 0.0, sizeof(float) * this->num_detections);
  memset(raw_bbox, 0.0,
         sizeof(float) * this->num_detections * this->num_keypoints);

  anchors = load_anchors(anchors_file);
}

PoseDetectionInterpreter::~PoseDetectionInterpreter() {
  if (nullptr != scores) {
    delete[] scores;
    scores = nullptr;
  }
  if (nullptr != raw_bbox) {
    delete[] raw_bbox;
    raw_bbox = nullptr;
  }
}

std::vector<float>
PoseDetectionInterpreter::load_anchors(char const *filename) {
  std::ifstream input_file;

  input_file.open(filename);
  if (!input_file.is_open()) {
    std::cerr << "Failed to open " << filename << "!\n";
    exit(-1);
  }

  std::vector<float> anchor_box;
  float box{0.0};
  while (true) {
    input_file >> box;
    anchor_box.push_back(box);
    if (input_file.eof() != 0)
      break;
  }
  input_file.close();
  return anchor_box;
}

void PoseDetectionInterpreter::decode_predictions(const float *raw_bbox,
                                                  const float *scores) {
  if (nullptr != raw_bbox && nullptr != scores) {
    memcpy(this->scores, scores, sizeof(float) * num_detections);
    memcpy(this->raw_bbox, raw_bbox,
           sizeof(float) * num_detections * num_keypoints);

    // Apply sigmoid to scores
    decode_scores();

    PoseDetection _detection;
    std::vector<PoseDetection> poses;

    // Filter scores > score_threshold
    for (size_t i{0}; i < num_detections; i++) {
      if (scores[i] > score_threshold) {
        _detection.set_score(scores[i]);
        _detection.set_bbox(decode_bbox(i));
        _detection.set_mid_hip_center(decode_mid_hip_center(i));
        _detection.set_full_body_size_rotation(
            decode_full_body_size_rotation(i));
        poses.push_back(_detection);
      }
    }

    // Filter IoU
    detected_poses = nms(poses, nms_threshold);
  }
}

// Apply sigmoid to scores
void PoseDetectionInterpreter::decode_scores() {
  for (size_t i{0}; i < num_detections; i++) {
    scores[i] = 1 / (1 + std::exp(-scores[i]));
  }
}

BoundingBox PoseDetectionInterpreter::decode_bbox(const size_t &index) {
  float centers_x, centers_y, sides_w, sides_h;

  BoundingBox bbox;

  if (index > num_detections) {
    std::cerr << "Got error index during bounding box decoding!\n";
    exit(-1);
  }

  // Decode bbox
  centers_x = anchors.data()[index * 4 + 0] +
              (raw_bbox[index * num_keypoints + 0] / scale);
  centers_y = anchors.data()[index * 4 + 1] +
              (raw_bbox[index * num_keypoints + 1] / scale);
  sides_w = raw_bbox[index * num_keypoints + 2] / scale;
  sides_h = raw_bbox[index * num_keypoints + 3] / scale;

  bbox["xmin"] = centers_x - sides_w / 2;
  bbox["ymin"] = centers_y - sides_h / 2;
  bbox["xmax"] = centers_x + sides_w / 2;
  bbox["ymax"] = centers_y + sides_h / 2;

  return bbox;
}

Keypoint PoseDetectionInterpreter::decode_mid_hip_center(const size_t &index) {
  float mid_hip_center_x, mid_hip_center_y;

  if (index > num_detections) {
    std::cerr << "Got error index during keypoint decoding!\n";
    exit(-1);
  }

  // Decode keypoint
  mid_hip_center_x = anchors.data()[index * 4 + 0] +
                     (raw_bbox[index * num_keypoints + 4] / scale);
  mid_hip_center_y = anchors.data()[index * 4 + 1] +
                     (raw_bbox[index * num_keypoints + 5] / scale);

  Keypoint kp(mid_hip_center_x, mid_hip_center_y);

  return kp;
}

Keypoint
PoseDetectionInterpreter::decode_full_body_size_rotation(const size_t &index) {
  float full_body_size_rotation_x, full_body_size_rotation_y;

  if (index > num_detections) {
    std::cerr << "Got error index during keypoint decoding!\n";
    exit(-1);
  }

  // Decode keypoint
  full_body_size_rotation_x = anchors.data()[index * 4 + 0] +
                              (raw_bbox[index * num_keypoints + 6] / scale);
  full_body_size_rotation_y = anchors.data()[index * 4 + 1] +
                              (raw_bbox[index * num_keypoints + 7] / scale);

  Keypoint kp(full_body_size_rotation_x, full_body_size_rotation_y);

  return kp;
}

std::vector<PoseDetection>
PoseDetectionInterpreter::nms(std::vector<PoseDetection> &poses,
                              const float &nms_threshold) {
  std::sort(poses.begin(), poses.end(), comparer);

  std::vector<PoseDetection> filtered_results;
  while (poses.size() > 0) {
    filtered_results.push_back(poses[0]);
    size_t index = 1;
    while (index < poses.size()) {
      float iou_value = iou(poses[0].get_bbox(), poses[index].get_bbox());
      if (iou_value > nms_threshold)
        poses.erase(poses.begin() + index);
      else
        index++;
    }
    poses.erase(poses.begin());
  }
  return filtered_results;
}

float PoseDetectionInterpreter::iou(const BoundingBox &bbox_a,
                                    const BoundingBox &bbox_b) {
  float x1 = std::max(bbox_a("xmin"), bbox_b("xmin"));
  float y1 = std::max(bbox_a("ymin"), bbox_b("ymin"));
  float x2 = std::min(bbox_a("xmax"), bbox_b("xmax"));
  float y2 = std::min(bbox_a("ymax"), bbox_b("ymax"));

  float w = std::max(static_cast<float>(0.0), (x2 - x1));
  float h = std::max(static_cast<float>(0.0), (y2 - y1));

  float inter = w * h;
  float areaA =
      (bbox_a("xmax") - bbox_a("xmin")) * (bbox_a("ymax") - bbox_a("ymin"));
  float areaB =
      (bbox_b("xmax") - bbox_b("xmin")) * (bbox_b("ymax") - bbox_b("ymin"));
  float o = inter / (areaA + areaB - inter);
  return (o >= 0) ? o : 0;
}

bool PoseDetectionInterpreter::comparer(PoseDetection &score_a,
                                        PoseDetection &score_b) {
  return score_a.get_score() > score_b.get_score();
}

std::vector<PoseDetection> PoseDetectionInterpreter::get_pose_detections() {
  return detected_poses;
}
