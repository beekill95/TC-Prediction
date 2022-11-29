import tensorflow as tf


def sst_loss(sst: tf.Tensor, sst_threshold=26 + 273.15):
    def _warm_sst_loss(y_true, y_pred):
        return (sst_threshold - sst) * y_pred

    return _warm_sst_loss
