import tensorflow as tf


def sst_loss(sst: tf.Tensor, sst_threshold=26 + 273.15):
    def _warm_sst_loss(y_true, y_pred):
        loss = tf.nn.relu(sst_threshold - sst) * y_pred
        return tf.reduce_mean(loss)

    return _warm_sst_loss
