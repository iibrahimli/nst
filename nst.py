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


def imshow(img, title=None):
    # get rid of batch dim
    out = np.squeeze(img, axis=0)
    print(out.dtype)
    out = out.astype(np.uint8)
    if title:
        plt.title(title)
    plt.imshow(out)


def load_and_process_img(path):
    img = load_img(path)

    # VGG 19 uses images normalized by mean = [103.939, 116.779, 123.68] and BGR channel order
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img


def deprocess_img(img):
    x = img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, axis=0)
    assert len(x.shape) == 3, ("Input to deprocess_img must be an image of dimension [1, height, width, channel] (or without the '1')")

    # add means to each channel
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # reverse order of channels
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype(np.uint8)
    return x


