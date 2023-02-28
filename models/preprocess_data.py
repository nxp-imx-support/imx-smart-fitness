#!/usr/bin/env python3

"""
Copyright © 2021 Patrick Levin
Copyright 2022-2023 NXP

SPDX-License-Identifier: MIT

Script to preprocess data used to calibrate pose_landmark_lite.tflite model.
Images are cropped and rotated using the predicted bboxes from pose_detection_quant.tflite model.

"""

import cv2
import numpy as np
import math
import tflite_runtime.interpreter as tflite

from glob import glob


# score limit is 100 in mediapipe and leads to overflows with IEEE 754 floats
# this lower limit is safe for use with the sigmoid functions and float32
RAW_SCORE_LIMIT = 80

# NMS similarity threshold
MIN_SUPPRESSION_THRESHOLD = 0.3


class PoseDetection(object):
    def __init__(self, model_path, threshold=0.5):
        # Define interpreter
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_index = self.interpreter.get_input_details()[0]["index"]
        self.input_shape = self.interpreter.get_input_details()[0]["shape"]
        self.bbox_index = self.interpreter.get_output_details()[1]["index"]
        self.score_index = self.interpreter.get_output_details()[0]["index"]

        # Load anchors for model
        self.anchors = np.load("../anchors.npy")
        self.threshold = threshold

    def _preprocess_img(self, input_data):
        # Normalize and pad image
        shape = np.r_[input_data.shape]
        pad = (shape.max() - shape[:2]).astype("uint32") // 2
        img_pad = np.pad(
            input_data, ((pad[0], pad[0]), (pad[1], pad[1]), (0, 0)), mode="constant"
        )
        img_pad = cv2.resize(img_pad, (256, 256))

        # Img for model
        img_norm = cv2.resize(img_pad, (224, 224))
        img_norm = np.ascontiguousarray(
            2 * ((np.ascontiguousarray(img_norm) / 255.0) - 0.5).astype("float32")
        )
        # Reshape tensors to fit input size of model
        img_norm = img_norm[np.newaxis, :, :, :]

        return img_norm, img_pad

    def detect(self, img):
        img_norm, img_pad = self._preprocess_img(img)

        self.interpreter.set_tensor(self.input_index, img_norm)
        self.interpreter.invoke()
        raw_boxes = self.interpreter.get_tensor(self.bbox_index)
        raw_scores = self.interpreter.get_tensor(self.score_index)

        boxes = self._decode_boxes(raw_boxes)
        scores = self._get_sigmoid_scores(raw_scores)

        score_above_threshold = scores > self.threshold
        filtered_boxes = boxes[np.argwhere(score_above_threshold)[:, 1], :]
        filtered_scores = scores[score_above_threshold]

        output_boxes = np.array(
            self._non_maximum_suppression(
                filtered_boxes, filtered_scores, MIN_SUPPRESSION_THRESHOLD
            )
        )
        return output_boxes, img_pad

    def _overlap_similarity(self, box1, box2):
        box1_tmp = box1[0:2, :].reshape(4)
        box2_tmp = box2[0:2, :].reshape(4)

        if box1_tmp is None or box2_tmp is None:
            return 0
        x1_min, y1_min, x1_max, y1_max = box1_tmp
        x2_min, y2_min, x2_max, y2_max = box2_tmp
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        x3_min = max(x1_min, x2_min)
        x3_max = min(x1_max, x2_max)
        y3_min = max(y1_min, y2_min)
        y3_max = min(y1_max, y2_max)
        intersect_area = (x3_max - x3_min) * (y3_max - y3_min)
        denominator = box1_area + box2_area - intersect_area
        return intersect_area / denominator if denominator > 0.0 else 0.0

    def _non_maximum_suppression(self, boxes, scores, min_suppression_threshold):
        candidates_list = []
        for i in range(np.size(boxes, 0)):
            candidates_list.append((boxes[i], scores[i]))
        candidates_list = sorted(candidates_list, key=lambda x: x[1], reverse=True)
        kept_list = []
        for sorted_boxes, sorted_scores in candidates_list:
            suppressed = False
            for kept in kept_list:
                similarity = self._overlap_similarity(kept, sorted_boxes)
                if similarity > min_suppression_threshold:
                    suppressed = True
                    break
            if not suppressed:
                kept_list.append(sorted_boxes)
        return kept_list

    def _decode_boxes(self, raw_boxes):
        """Simplified version of
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
        """
        # width == height so scale is the same across the board
        scale = self.input_shape[1]
        num_points = raw_boxes.shape[-1] // 2

        # scale all values (applies to positions, width, and height alike)
        boxes = raw_boxes.reshape(-1, num_points, 2) / scale
        # adjust center coordinates and key points to anchor positions
        boxes[:, 0] += self.anchors[:, :2]
        for i in range(2, num_points):
            boxes[:, i] += self.anchors[:, :2]
        # convert x_center, y_center, w, h to xmin, ymin, xmax, ymax
        center = np.array(boxes[:, 0])
        half_size = boxes[:, 1] / 2
        boxes[:, 0] = center - half_size
        boxes[:, 1] = center + half_size

        return boxes

    def _get_sigmoid_scores(self, raw_scores):
        raw_scores[raw_scores < -RAW_SCORE_LIMIT] = -RAW_SCORE_LIMIT
        raw_scores[raw_scores > RAW_SCORE_LIMIT] = RAW_SCORE_LIMIT
        return 1 / (1 + np.exp(-raw_scores))


if __name__ == "__main__":
    # Threshold value for pose detection
    detect_threshold = 0.5

    # Load images
    img_list = sorted(glob("../data/coco_calib_data/*"))

    # Construct pose detection
    detector = PoseDetection("../pose_detection_quant.tflite", detect_threshold)

    for idx in range(len(img_list)):
        frame = cv2.imread(img_list[idx], cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect pose using pose detection model
        detections, frame = detector.detect(frame)

        if np.size(detections, 0) > 0:
            boxes = detections[:, 0:2, :].reshape(-1, 4)
            mid_hip_center = detections[:, 2:3, :].reshape(-1, 2)
            full_body_sr = detections[:, 3:4, :].reshape(-1, 2)

            for i in range(np.size(boxes, 0)):
                boxes[i][[0, 2]] *= frame.shape[1]
                boxes[i][[1, 3]] *= frame.shape[0]
                mid_hip_center[i][[0]] *= frame.shape[1]
                mid_hip_center[i][[1]] *= frame.shape[0]
                full_body_sr[i][0] *= frame.shape[1]
                full_body_sr[i][1] *= frame.shape[0]

            mid_hip_center = mid_hip_center.astype(np.int32)
            full_body_sr = full_body_sr.astype(np.int32)

            for i in range(np.size(boxes, 0)):
                x_mid_hip_center, y_mid_hip_center = mid_hip_center[i]
                x_full_body_sr, y_full_body_sr = full_body_sr[i]

                degrees = math.degrees(
                    math.atan2(
                        y_mid_hip_center - y_full_body_sr,
                        x_mid_hip_center - x_full_body_sr,
                    )
                )

                distance = math.sqrt(
                    pow(x_full_body_sr - x_mid_hip_center, 2)
                    + pow(y_full_body_sr - y_mid_hip_center, 2)
                )

                # Obtain bbox coordinates
                xmin = int(x_mid_hip_center - distance)
                ymin = int(y_mid_hip_center - distance)
                xmax = int(x_mid_hip_center + distance)
                ymax = int(y_mid_hip_center + distance)

                # Create bbox array
                bbox = np.array(
                    ((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax))
                )

                # Rotate image to align with pose landmarks model
                rotMat = cv2.getRotationMatrix2D(
                    (int(x_mid_hip_center), int(y_mid_hip_center)), degrees - 90, 1.0
                )

                # Not rotated, but origin is re-calculated for new rotated image
                rot_mat_bbox = cv2.getRotationMatrix2D(
                    (int(x_mid_hip_center), int(y_mid_hip_center)), 0, 1.0
                )

                bbox_rotated = np.vstack((bbox.T, np.array((1, 1, 1, 1))))
                bbox_rotated = np.int0(np.dot(rot_mat_bbox, bbox_rotated).T)
                frame_rotated = cv2.warpAffine(frame, rotMat, frame.shape[1::-1])

                # Pad rotated image to avoid bbox out of range
                frame_rotated = np.pad(
                    frame_rotated, ((200, 200), (200, 200), (0, 0)), mode="constant"
                )

                detected_pose = frame_rotated[
                    bbox_rotated[1][1] + 200 : bbox_rotated[3][1] + 200,
                    bbox_rotated[0][0] + 200 : bbox_rotated[2][0] + 200,
                ]

                print("Preprocessed image: ", idx)
                detected_pose = cv2.cvtColor(detected_pose, cv2.COLOR_BGR2RGB)
                cv2.imwrite(
                    "../data/coco_calib_data_cropped/img" + str(idx) + ".jpg",
                    detected_pose,
                )
