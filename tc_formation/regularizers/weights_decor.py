import tensorflow as tf
import tensorflow.keras.regularizers as regularizers

from .utils import corr_coef


class WeightsCorrRegularizer(regularizers.Regularizer):
    def __init__(self, strength: float) -> None:
        super().__init__()

        self._half_strength = strength / 2.

    def __call__(self, filters):
        # filters should have shape (H, W, in_channels, out_channels)
        out_channels = tf.shape(filters)[-1]
        filters = tf.reshape(filters, (-1, out_channels))
        corr = corr_coef(filters)
        return self._half_strength * tf.norm(corr, ord='fro', axis=[-2, -1])

    def get_config(self):
        return dict(strength=self._half_strength * 2.)
