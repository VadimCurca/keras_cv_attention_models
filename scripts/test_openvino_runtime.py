import openvino.runtime as ov
from openvino.runtime import Core, Layout, Type
from openvino.preprocess import PrePostProcessor, ResizeAlgorithm

import numpy as np
import tensorflow as tf
from tensorflow import keras
from skimage.data import chelsea
from PIL import Image as im

core = ov.Core()

model_xml_path = "/home/cvadim/learning/ML/keras_cv_attention_models/saved_models/openvino-model/saved_model.xml"
model = core.read_model(model_xml_path)

imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
imm = tf.image.resize(imm, (224, 224))

pil_img = tf.keras.preprocessing.image.array_to_img(imm)
pil_img.save('gfg_dummy_pic.png')

# Set batch size = 1
input_array = np.expand_dims(imm.numpy(), 0)

# Assume intput is NHWC
_, h, w, _ = input_array.shape

print("input shape = ", input_array.shape)      # -> (1, 224, 224, 3)
print("input data type = ", input_array.dtype)  # -> float32

ppp = PrePostProcessor(model)

# Set input tensor information:
ppp.input().tensor() \
    .set_shape(input_array.shape) \
    .set_element_type(Type.f32) \
    .set_layout(Layout('NHWC'))

# Set model input information:
ppp.input().preprocess().resize(ResizeAlgorithm.RESIZE_LINEAR)
ppp.input().model().set_layout(Layout('NHWC'))
ppp.output().tensor().set_element_type(Type.f32)
model = ppp.build()

# Compile model for specific device (same as compile_tool app)
compiled_model = core.compile_model(model, "CPU")

results = compiled_model.infer_new_request({0: input_array})

predictions = next(iter(results.values()))

print(keras.applications.imagenet_utils.decode_predictions(predictions))

