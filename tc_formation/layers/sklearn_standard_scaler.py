import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


class SklearnStandardScaler(tf.keras.layers.Layer):
    def __init__(self, scaler: StandardScaler) -> None:
        super().__init__()

        # Init non trainable weight.
        self.means = tf.Variable(scaler.mean_, trainable=False)
        self.stds = tf.Variable(np.sqrt(scaler.var_), trainable=False)

    def call(self, inputs):
        return (inputs - self.means) / self.stds


class SklearnStandardScalerInverse(tf.keras.layers.Layer):
    def __init__(self, scaler: StandardScaler) -> None:
        super().__init__()

        # Init non trainable weight.
        self.means = tf.Variable(scaler.mean_, trainable=False)
        self.stds = tf.Variable(np.sqrt(scaler.var_), trainable=False)

    def call(self, inputs):
        return inputs * self.stds + self.means

