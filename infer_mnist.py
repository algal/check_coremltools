#!/usr/bin/env python
import coremltools
from PIL import Image
import numpy as np
import sys

print(sys.argv)
if len(sys.argv) > 1:
    model_path = sys.argv[1]
else:
    model_path = "./MNISTClassifier.mlmodel"

try:
    print("coremltools versison: {}".format(coremltools.__version__))
except:
    print("coremltools imported byt is not properly installed")
    exit(1)

# im = Image.effect_mandelbrot((224,224),(-3, -2.5, 2, 2.5),100)
# data = np.array(Image.new('RGB', (224, 224), (228, 150, 150)))
# #data = np.empty((224,224,3), dtype=np.uint8)
# input_image = Image.fromarray(data,'RGB')
input_image = Image.new('RGB', (224, 224), (228, 150, 150))

model = coremltools.models.MLModel(model_path)
try:
    print(model.predict({'image': input_image}))
    print("successfully ran CoreML inference")
except:
    print("Error running CoremL inference")

