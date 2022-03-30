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


class FromLogitsDecorator(tf.metrics.Metric):
    def __init__(self, metric, *args, **kwargs):
        super().__init__(*args, name=metric.name, **kwargs)
        self._metric = metric

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.nn.sigmoid(y_pred)
        return self._metric.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self._metric.result()



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
    def __init__(self, name='f1_score', class_id=None, from_logits=False, **kwargs):
        super().__init__(name=name, **kwargs)
        # self.precision_fn = tf.metrics.Precision(thresholds=0.5, class_id=class_id, **metric_kwargs)
        # self.recall_fn = tf.metrics.Recall(thresholds=0.5, class_id=class_id, **metric_kwargs)
        self.precision_fn = PrecisionScore(thresholds=0.5, class_id=class_id, from_logits=from_logits)
        self.recall_fn = RecallScore(thresholds=0.5, class_id=class_id, from_logits=from_logits)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision_fn.update_state(y_true, y_pred)
        self.recall_fn.update_state(y_true, y_pred)

    def result(self):
        p = self.precision_fn.result()
        r = self.recall_fn.result()
        return 2 * p * r / (p + r + 1e-6)

    def reset_states(self):
        # we also need to reset the state of the precision and recall objects
        self.precision_fn.reset_states()
        self.recall_fn.reset_states()
