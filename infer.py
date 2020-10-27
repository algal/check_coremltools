#!/usr/bin/env python
import coremltools as ct
from PIL import Image
import numpy as np
import sys
from pathlib import Path

"""
Runs inference on an MLModel which takes a single RGB image.

Resizes the image to fit the model's required image dimensions
"""

if len(sys.argv) != 3:
    print(f"{sys.argv[0]} modelpath imagepath")
    print("")
    print("  modelpath - path to an MLModel which takes one RGB image")
    print("  imagepath - path to a an RGB image")
    exit(1)
model_path = Path(sys.argv[1])
image_path = Path(sys.argv[2])

try:
    print("coremltools versison: {}".format(ct.__version__))
except:
    print("coremltools is not properly installed")
    exit(1)

if model_path.exists() is False:
    print(f"model does not exist at {model_path}")
    exit(1)
    
if image_path.exists() is False:
    print(f"image does not exist at {image_path}")
    exit(1)

model = ct.models.MLModel(str(model_path))

assert len(model._spec.description.input) == 1, "not exactly 1 input feature"

input_feature = model._spec.description.input[0]

W = input_feature.type.imageType.width
H = input_feature.type.imageType.height

from coremltools.proto import FeatureTypes_pb2 as _FeatureTypes_pb2
colorSpace = _FeatureTypes_pb2.ImageFeatureType.ColorSpace.Name( input_feature.type.imageType.colorSpace )

assert colorSpace == 'RGB', "model does not take RGB images"

image = Image.open(str(image_path)).resize( (W,H) )

assert image.mode == 'RGB', "image is not an RGB image"

feature_name = input_feature.name

out = model.predict({feature_name: image})

try:
    print(model.predict({feature_name: image}))
    print("successfully ran CoreML inference")
except:
    print("Error running CoremL inference")

