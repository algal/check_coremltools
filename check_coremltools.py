#!/usr/bin/env python
import coremltools
from PIL import Image
import numpy as np

try:
    print("coremltools versison: {}".format(coremltools.__version__))
except:
    print("coremltools imported byt is not properly installed")
    exit(1)

data = np.empty((28,28), dtype=np.uint8)
input_image = Image.fromarray(data)

model = coremltools.models.MLModel('./MNISTClassifier.mlmodel')
print(model.predict({'image': input_image}))
