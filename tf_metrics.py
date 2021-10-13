"""
A workaround for working with metrics that don't support `from_logits=True` output.
Copied from: https://github.com/tensorflow/tensorflow/issues/42182#issuecomment-818777681
"""
import tensorflow as tf
import tensorflow_addons as tfa


class FromLogitsMixin:
    def __init__(self, from_logits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.from_logits = from_logits

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
        return super().update_state(y_true, y_pred, sample_weight)


class F1Score(FromLogitsMixin, tfa.metrics.F1Score):
    ...


class PrecisionScore(FromLogitsMixin, tf.metrics.Precision):
    ...


class RecallScore(FromLogitsMixin, tf.metrics.Recall):
    ...
