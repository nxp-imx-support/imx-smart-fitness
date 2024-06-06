# MediaPipe Models

Two [MediaPipe](https://developers.google.com/mediapipe) models are used to obtain the pose estimation:

* BlazePose Detector
* BlazePose GHUM 3D Lite

## Download and quantize models for i.MX NPU acceleration

To create the fully quantized INT8 models with FLOAT32 input and FLOAT32
output, run:

```bash
bash recipe.sh
```

>**NOTE:** When running this script, please make sure you are not using the i.MX Toolchain for
cross-compilatio, otherwise the process will fail. This script compiles for x86 Host-PC.

This script requires `bash` shell and the following installed packages: `cmake`, `pip` and `miniconda` or `anaconda` for python version management. Please make sure you have these dependencies
installed in you system.

It was tested on the following OS:

* Ubuntu 20.04
* Ubuntu 22.04

After `recipe.sh` finishes, the TensorFlow Lite models are located at
`./deploy/pose_detection_quant.tflite` and `./deploy/pose_landmark_lite_quant.tflite`.

## Models information

### BlazePose Detector

Information          | Value
---                  | ---
Input shape          | RGB image [1, 224, 224, 3]
Input value range    | [-1.0, 1.0]
Output shape         | Undecoded face bboxes location and keypoints: [1, 2254, 12] <br /> Scores of detected bboxes: [1,2254,1]
MACs                 | 433.535 M
File size (INT8)     | 3.5 MB
Source framework     | MediaPipe (TensorFlow Lite)
Target platform      | MPUs

### BlazePose GHUM 3D Lite

Information          | Value
---                  | ---
Input shape          | RGB image [1, 256, 256, 3]
Input value range    | [0.0, 1.0]
Output shape         | Pose Landmarks: [1, 195] <br /> Presence of pose: [1, 1] <br /> Segmentation mask for pose: [1, 256, 256, 1] <br /> Heat map for pose: [1, 64, 64, 39] <br /> World landmarks for pose: [1, 117]
MACs                 | 202.980 M
File size (INT8)     | 1.8 MB
Source framework     | MediaPipe (TensorFlow Lite)
Target platform      | MPUs

## Benchmarks

The quantized INT8 models have been tested on i.MX 8M Plus and i.MX 93 using `./benchmark_model` tool
(see [i.MX Machine Learning User's Guide](https://www.nxp.com/docs/en/user-guide/IMX-MACHINE-LEARNING-UG.pdf)).

### BlazePose performance

Platform    | Accelerator     | Avg. latency | Command
---         | ---             | ---          | ---
i.MX 8M Plus | CPU (1 thread)  | 169.48 ms    | ./benchmark_model --graph=pose_detection_quant.tflite
i.MX 8M Plus | CPU (4 threads) |  65.60 ms    | ./benchmark_model --graph=pose_detection_quant.tflite --num_threads=4
i.MX 8M Plus | NPU             |   8.04 ms    | ./benchmark_model --graph=pose_detection_quant.tflite --external_delegate_path=/usr/lib/libvx_delegate.so
i.MX 93      | CPU (1 thread)  | 104.16 ms    | ./benchmark_model --graph=pose_detection_quant.tflite
i.MX 93      | CPU (2 threads) |  71.06 ms    | ./benchmark_model --graph=pose_detection_quant.tflite --num_threads=2
i.MX 93      | NPU             |   7.33 ms    | ./benchmark_model --graph=pose_detection_quant_vela.tflite --external_delegate_path=/usr/lib/libethosu_delegate.so

### BlazePose GHUM 3D Lite performance

Platform    | Accelerator     | Avg. latency | Command
---         | ---             | ---          | ---
i.MX 8M Plus | CPU (1 thread)  | 160.75 ms    | ./benchmark_model --graph=pose_landmark_lite_quant.tflite
i.MX 8M Plus | CPU (4 threads) |  77.81 ms    | ./benchmark_model --graph=pose_landmark_lite_quant.tflite --num_threads=4
i.MX 8M Plus | NPU             |  16.13 ms    | ./benchmark_model --graph=pose_landmark_lite_quant.tflite --external_delegate_path=/usr/lib/libvx_delegate.so
i.MX 93      | CPU (1 thread)  | 116.50 ms    | ./benchmark_model --graph=pose_landmark_lite_quant.tflite
i.MX 93      | CPU (2 threads) |  84.83 ms    | ./benchmark_model --graph=pose_landmark_lite_quant.tflite --num_threads=2
i.MX 93      | NPU             |  10.05 ms     | ./benchmark_model --graph=pose_landmark_lite_quant_vela.tflite --external_delegate_path=/usr/lib/libethosu_delegate.so

>**NOTE:** Evaluated on BSP LF-6.6.23_2.0.0.

## Origin

[1] BlazePose: On-device Real-time Body Pose tracking, CVPR Workshop on Computer Vision for Augmented and Virtual
Reality, Seattle, WA, USA, 2020.

[2] GHUM & GHUML: Generative 3D Human Shape and Articulated Pose Models Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition, pages 6184-6193, 2020.

Models created by: Valentin Bazarevsky, Google; Ivan Grishchenko, Google; Eduard Gabriel Bazavan, Google.

Model card: https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf

## Licensing

MediaPipe models are licensed under [Apache-2.0 License](https://www.apache.org/licenses/LICENSE-2.0.html).
