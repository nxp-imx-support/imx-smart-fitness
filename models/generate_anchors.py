#!/usr/bin/env python3

# Copyright 2019 Matthijs Hollemans
# Copyright 2023 NXP
#
# SPDX-License-Identifier: Apache-2.0
#
# This script is based on the following repository: https://github.com/hollance/BlazeFace-PyTorch
# This work is licensed under the same terms as MediaPipe (Apache-2.0 License)

import numpy as np

pose_detection = {
    "num_layers": 5,
    "min_scale": 0.1484375,
    "max_scale": 0.75,
    "input_size_height": 224,
    "input_size_width": 224,
    "anchor_offset_x": 0.5,
    "anchor_offset_y": 0.5,
    "strides": [8, 16, 32, 32, 32],
    "aspect_ratios": [1.0],
    "reduce_boxes_in_lowest_layer": False,
    "interpolated_scale_aspect_ratio": 1.0,
    "fixed_anchor_size": True,
}


def calculate_scale(min_scale, max_scale, stride_index, num_strides):
    return min_scale + (max_scale - min_scale) * stride_index / (num_strides - 1.0)


def generate_anchors(options):
    strides_size = len(options["strides"])
    assert options["num_layers"] == strides_size

    anchors = []
    layer_id = 0
    while layer_id < strides_size:
        anchor_height = []
        anchor_width = []
        aspect_ratios = []
        scales = []

        # For same strides, we merge the anchors in the same order.
        last_same_stride_layer = layer_id
        while (last_same_stride_layer < strides_size) and (
            options["strides"][last_same_stride_layer] == options["strides"][layer_id]
        ):
            scale = calculate_scale(
                options["min_scale"],
                options["max_scale"],
                last_same_stride_layer,
                strides_size,
            )

            if last_same_stride_layer == 0 and options["reduce_boxes_in_lowest_layer"]:
                # For first layer, it can be specified to use predefined anchors.
                aspect_ratios.append(1.0)
                aspect_ratios.append(2.0)
                aspect_ratios.append(0.5)
                scales.append(0.1)
                scales.append(scale)
                scales.append(scale)
            else:
                for aspect_ratio in options["aspect_ratios"]:
                    aspect_ratios.append(aspect_ratio)
                    scales.append(scale)

                if options["interpolated_scale_aspect_ratio"] > 0.0:
                    scale_next = (
                        1.0
                        if last_same_stride_layer == strides_size - 1
                        else calculate_scale(
                            options["min_scale"],
                            options["max_scale"],
                            last_same_stride_layer + 1,
                            strides_size,
                        )
                    )
                    scales.append(np.sqrt(scale * scale_next))
                    aspect_ratios.append(options["interpolated_scale_aspect_ratio"])

            last_same_stride_layer += 1

        for i in range(len(aspect_ratios)):
            ratio_sqrts = np.sqrt(aspect_ratios[i])
            anchor_height.append(scales[i] / ratio_sqrts)
            anchor_width.append(scales[i] * ratio_sqrts)

        stride = options["strides"][layer_id]
        feature_map_height = int(np.ceil(options["input_size_height"] / stride))
        feature_map_width = int(np.ceil(options["input_size_width"] / stride))

        for y in range(feature_map_height):
            for x in range(feature_map_width):
                for anchor_id in range(len(anchor_height)):
                    x_center = (x + options["anchor_offset_x"]) / feature_map_width
                    y_center = (y + options["anchor_offset_y"]) / feature_map_height

                    new_anchor = [x_center, y_center, 0, 0]
                    if options["fixed_anchor_size"]:
                        new_anchor[2] = 1.0
                        new_anchor[3] = 1.0
                    else:
                        new_anchor[2] = anchor_width[anchor_id]
                        new_anchor[3] = anchor_height[anchor_id]
                    anchors.append(new_anchor)

        layer_id = last_same_stride_layer

    return anchors


# Save anchors
anchors = generate_anchors(pose_detection)
np.save("../anchors.npy", anchors)
np.savetxt("../anchors.txt", anchors)
