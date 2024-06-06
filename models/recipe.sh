#!/usr/bin/env bash
# Copyright 2023-2024 NXP
# SPDX-License-Identifier: Apache-2.0

# Exit immediately if a command exits with a non-zero status
set -e

# Clean environment before running recipe
rm -rf tmp/ deploy/ *.tflite *.zip data/coco_calib_data*

# fetch images from COCO dataset
wget http://images.cocodataset.org/zips/val2017.zip -O val2017.zip
unzip val2017.zip

# Filter images to only contain human poses
mkdir -p coco_calib_data
for file in $(cat data/coco_calib_images.txt); do mv "$file" coco_calib_data; done
rm -rf val2017.zip val2017
mv coco_calib_data data/

# Download Pose Detection model and Pose Landmark model from MediaPipe's webpage
mkdir -p tmp && cd tmp
wget https://storage.googleapis.com/mediapipe-assets/pose_detection.tflite
wget https://storage.googleapis.com/mediapipe-assets/pose_landmark_lite.tflite
cd ../ && mv tmp/pose_detection.tflite .
mv tmp/pose_landmark_lite.tflite .
rm -rf tmp

CONDA_ENVIRONMENT=$(conda info | grep -i 'base environment' | awk '{print $4}')
echo ${CONDA_ENVIRONMENT}
source ${CONDA_ENVIRONMENT}/etc/profile.d/conda.sh

conda create --name eiq-model-zoo-env python=3.8.18 --yes
conda activate eiq-model-zoo-env

# Create folder for model conversion and quantization
mkdir tmp
(
	cd tmp

	# Clone flatbuffers repo and build it
	git clone -b v2.0.8 https://github.com/google/flatbuffers.git
	cd flatbuffers && mkdir build && cd build
	cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release ..
	make -j$(nproc)

	# Download schema.fbs
	cd ../../
	wget https://github.com/PINTO0309/tflite2tensorflow/raw/main/schema/schema.fbs

	# MediaPipe *.tflite -> TensorFlow (*.pb)
	python3 -m venv env_tflite2tf
	source ./env_tflite2tf/bin/activate
	pip install --upgrade pip
	pip install pandas
	pip install --upgrade git+https://github.com/PINTO0309/tflite2tensorflow

	APPVER=v1.20.7
	TENSORFLOWVER=2.8.0

	# Install the customized full TensorFlow package (MediaPipe Custom OP, FlexDelegate, XNNPACK enabled)
	wget https://github.com/PINTO0309/tflite2tensorflow/releases/download/${APPVER}/tensorflow-${TENSORFLOWVER}-cp38-none-linux_x86_64.whl &&
		pip3 install --force-reinstall tensorflow-${TENSORFLOWVER}-cp38-none-linux_x86_64.whl &&
		rm tensorflow-${TENSORFLOWVER}-cp38-none-linux_x86_64.whl

	# Fix protobuf version
	pip install protobuf==3.20

	# Convert MediaPipe Model to TF (*.pb)
	cd ..
	tflite2tensorflow --model_path pose_detection.tflite --flatc_path ./tmp/flatbuffers/build/flatc --schema_path ./tmp/schema.fbs --output_pb
	mv saved_model pose_detection_saved_model
	tflite2tensorflow --model_path pose_landmark_lite.tflite --flatc_path ./tmp/flatbuffers/build/flatc --schema_path ./tmp/schema.fbs --output_pb
	mv saved_model pose_landmark_saved_model

	deactivate
)

# Remove tmp folder
rm -rf tmp
rm pose_detection.json
rm pose_landmark_lite.json

# Create folder for model quantization
mkdir tmp
(
	# TensorFlow (*.pb) -> TFLite (*.tflite)
	cd tmp
	python3 -m venv env_tf
	source ./env_tf/bin/activate

	pip install --upgrade pip
	pip install -r ../requirements_tf.txt

	# Generate anchors for pose_detection.tflite model
	python3 ../generate_anchors.py

	python -c '
import tensorflow as tf
import numpy as np

import cv2
from glob import glob
import random

random.seed(1337)

img_list = sorted(glob("../data/coco_calib_data/*"))

random.shuffle(img_list)


def _normalize(input_data):
    input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
    shape = np.r_[input_data.shape]
    pad = (shape.max() - shape[:2]).astype("uint32") // 2
    img_pad = np.pad(
                input_data, ((pad[0],pad[0]), (pad[1],pad[1]), (0,0)), mode="constant")

    img_small = cv2.resize(img_pad, (224, 224))
    img_small = np.ascontiguousarray(img_small)
    img = np.ascontiguousarray(2 * ((np.asarray(img_small) / 255.0) - 0.5).astype("float32"))
    return img


def representative_data_gen():
    for i in range(len(img_list)):
        img = cv2.imread(img_list[i], cv2.IMREAD_COLOR)
        img = _normalize(img)
        img = img[None, ...]
        yield [img]


# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model("../pose_detection_saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.representative_dataset = representative_data_gen
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32
tflite_model = converter.convert()

# Save the model.
with open("../pose_detection_quant.tflite", "wb") as f:
    f.write(tflite_model)    
'

	mkdir ../data/coco_calib_data_cropped
	python3 ../preprocess_data.py

	python -c '
import tensorflow as tf
import numpy as np

from PIL import Image
from glob import glob
import random

random.seed(1337)

img_list = sorted(glob("../data/coco_calib_data_cropped/*"))

random.shuffle(img_list)


def _normalize(img):
    img = np.ascontiguousarray(np.asarray(img) / 255.0).astype("float32")
    return img


def representative_data_gen():
    for i in range(len(img_list)):
        img = Image.open(img_list[i]).convert("RGB")
        img = img.resize((256, 256))
        img = _normalize(img)
        img = img[None, ...]
        yield [img]


# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model("../pose_landmark_saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.representative_dataset = representative_data_gen
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32
tflite_model = converter.convert()

# Save the model.
with open("../pose_landmark_lite_quant.tflite", "wb") as f:
    f.write(tflite_model)    
'
)

conda deactivate
conda remove -n eiq-model-zoo-env --all --yes

# Remove tmp and extra resources
rm -rf pose_detection_saved_model
rm -rf pose_landmark_saved_model
rm -rf tmp
rm pose_detection.tflite
rm pose_landmark_lite.tflite
rm anchors.npy

# Create deploy folder
mkdir deploy
mv *.tflite anchors.txt deploy/
cp pose_embeddings.csv deploy/
