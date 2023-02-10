import numpy as np
import tensorflow as tf


class SklearnPCALayer(tf.keras.layers.Layer):
    def __init__(self, components: np.ndarray) -> None:
        """
        Init this layer from component vectors given by sklearn's PCA.
        """
        super().__init__()

        # `components` should have shape (n_components, C)
        # thus `components_T` will have shape (C, n_components)
        self.components_T = tf.Variable(components.T, trainable=False, dtype=tf.float32)

    def call(self, inputs):
        # `inputs` is of shape (B, W, H, C)
        return tf.einsum('bwhc,cn->bwhn', inputs, self.components_T)

    def get_config(self):
        return dict(components=self.components_T.numpy().T)
