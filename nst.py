import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import functools

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers


# enable eager execution
tf.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))


def load_img(path):
    max_dim = 512
    img = Image.open(path)
    long_dim = max(img.size)
    scale = max_dim / long_dim
    
    # resize the image so that long side is 512 pixels
    img = img.resize((round(img.size[0]*scale), round(img[1]*scale)), Image.ANTIALIAS)
    img = kp_image.img_to_array(img)

    # add a batch dimension
    img = np.expand_dims(img, axis=0)

    return img



