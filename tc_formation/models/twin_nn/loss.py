import tensorflow as tf


def inner_distance_loss(y_pred: tf.Tensor):
    return tf.cond(tf.shape(y_pred)[0] == 0,
                   lambda: tf.constant(1e-3),
                   lambda: tf.reduce_mean(tf.square(y_pred)))
    # return tf.reduce_mean(tf.square(y_pred))


def outer_distance_loss(y_true, y_pred: tf.Tensor):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.tanh(y_pred)
    return tf.cond(tf.shape(y_pred)[0] == 0,
                   lambda: tf.constant(1e-3),
                   lambda: tf.reduce_mean(tf.square(y_true - y_pred)))
    # return tf.reduce_mean(tf.square(y_true - y_pred))


class TwinNNLoss:
    def __init__(self, label: int, C: float = 1.0) -> None:
        self._label = label
        self._C = C

    def __call__(self, y_true, y_pred: tf.Tensor) -> tf.Tensor:
        current_label_mask = y_true == self._label
        inner_dist = inner_distance_loss(y_pred[current_label_mask])
        # inner_dist = 0
        outer_dist = outer_distance_loss(
                y_true[~current_label_mask],
                y_pred[~current_label_mask])

        return 0.5 * (outer_dist + self._C * inner_dist)
