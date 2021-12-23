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


class NthClassificationMixin:
    def __init__(self, nth=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nth = nth

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true[:, :self._nth], y_pred[:, :self._nth], sample_weight)


class NthF1Score(NthClassificationMixin, F1Score):
    ...


class NthPrecisionScore(NthClassificationMixin, PrecisionScore):
    ...


class NthRecallScore(NthClassificationMixin, RecallScore):
    ...


class NthBinaryAccuracy(NthClassificationMixin, tf.metrics.BinaryAccuracy):
    ...


class CustomF1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', class_id=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='f1', initializer='zeros')
        self.precision_fn = tf.metrics.Precision(thresholds=0.5, class_id=class_id)
        self.recall_fn = tf.metrics.Recall(thresholds=0.5, class_id=class_id)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision_fn.update_state(y_true, y_pred)
        self.recall_fn.update_state(y_true, y_pred)

        p = self.precision_fn.result()
        r = self.recall_fn.result()

        # since f1 is a variable, we use assign
        self.f1.assign(2 * p * r / (p + r + 1e-6))

    def result(self):
        return self.f1

    def reset_states(self):
        # we also need to reset the state of the precision and recall objects
        self.precision_fn.reset_states()
        self.recall_fn.reset_states()
        self.f1.assign(0)
