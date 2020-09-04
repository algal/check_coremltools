#!/usr/bin/env python
import coremltools
from PIL import Image
import numpy as np

model_path = "./MNISTClassifier.mlmodel"

try:
    print("coremltools versison: {}".format(coremltools.__version__))
except:
    print("coremltools imported byt is not properly installed")
    exit(1)

data = np.empty((28,28), dtype=np.uint8)
input_image = Image.fromarray(data)
model = coremltools.models.MLModel(model_path)

try:
    print("Running CoreML inference on MNIST")
    print(model.predict({'image': input_image}))
    print("Ran CoreML inference")
except:
    print("Exception while running CoreML inference")


