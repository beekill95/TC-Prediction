import tensorflow as tf
import tensorflow.keras.regularizers as regularizers

from .utils import corr_coef, cov


class ActivationDeCovRegularizer(regularizers.Regularizer):
    def __init__(self, strength: float, use_corr: bool = False) -> None:
        super().__init__()

        self._half_strength = strength / 2.
        self._use_corr = use_corr

    def __call__(self, x):
        # x should have shape (None, nb_features)
        # basically, x is the output of a dense layer.
        x = tf.transpose(x)
        matrix = corr_coef(x) if self._use_corr else cov(x)
        diag = tf.linalg.diag_part(matrix)
        return self._half_strength * (
            tf.norm(matrix, ord='fro', axis=[-2, -1]) - tf.norm(diag))

    def get_config(self):
        return dict(
            strength=self._half_strength * 2.,
            use_corr=self._use_corr)
