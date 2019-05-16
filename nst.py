import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import functools

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tensorflow.keras.preprocessing import image as kp_image
from tensorflow.keras import models
from tensorflow.keras import losses
from tensorflow.keras import layers


# enable eager execution
tf.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))
