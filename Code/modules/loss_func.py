# Cutomized Loss Function for Age and Height

import tensorflow as tf
import numpy as np

def custom_loss_function(y_true, y_pred):
    weights = np.array([0.7, 0.3])
    squared_difference = tf.square(y_true - y_pred) * weights
    return tf.reduce_mean(squared_difference, axis=-1)