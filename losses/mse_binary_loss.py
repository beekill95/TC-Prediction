import tensorflow as tf
import tensorflow_addons as tfa


def mse_binary_crossentropy_loss(y_true, y_pred, alpha=1):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # MSE part.
    mse = tf.reduce_mean(tf.math.squared_difference(
        y_pred[:, 1:], y_true[:, 1:]), axis=-1)

    # Classification part.
    cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=y_pred[:, 0],
        labels=y_true[:, 0])

    return alpha * cross_entropy_loss + y_true[:, 0] * mse


def mse_focal_loss(y_true, y_pred, alpha=1):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    # MSE loss.
    mse = tf.reduce_mean(tf.math.squared_difference(
        y_pred[:, 1:], y_true[:, 1:]), axis=-1)

    # Classification loss.
    fl = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True)
    focal_loss = fl(y_true=y_true[:, :1], y_pred=y_pred[:, :1])

    return alpha * focal_loss + y_true[:, 0] * mse
