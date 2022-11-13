import sys
sys.path.append('..')

from keras_cv_attention_models import davit
import tensorflow as tf
from tensorflow import keras
from skimage.data import chelsea

# Will download and load pretrained imagenet weights.
mm = davit.DaViT_T(pretrained="imagenet")
mm.save('saved_model/DaVit-SavedModel-format2')

# Run prediction
imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
