from functools import wraps
import tensorflow as tf

# TODO: what should I do in cases having no positive samples in an image?
def hard_negative_mining(loss_func):
    """
    A decorator of a loss function `loss_func` to perform hard negative mining with
    negative to positive samples indicated by `negative_ratio`.

    The function also assumes that it is being applied to binary classification problem.
    Also, the `loss_func` should return loss tensor with shape [batch_size, height, width].

    Hard negative mining is implemented as noted (with some modifications) in
    [SSD: Single Shot Multibox Detector](https://arxiv.org/abs/1512.02325).
    """
    @wraps(loss_func)
    def hard_negative_mined_loss_fn(y_true, y_pred, negative_ratio=3, lampda=10, *args, **kwargs):
        y_true_ = tf.squeeze(y_true, axis=-1)
        _, height, width = y_true_.shape

        # Calculate the loss of the binary classification problem.
        # The loss will be of shape [batch_size, height, width, 1]
        loss = loss_func(y_true, y_pred, *args, **kwargs)
        # print(loss.shape)

        # Calculate how many negative samples we should take.
        positive_mask = y_true_ > 0
        nb_positives = tf.reduce_sum(tf.cast(positive_mask, dtype=tf.int32), axis=[1, 2])
        avg_positives = tf.reduce_mean(nb_positives[nb_positives > 0])
        nb_negatives = avg_positives * negative_ratio
        # tf.print('positives', nb_positives[nb_positives > 0])
        # tf.print('nb positives', avg_positives, 'nb negatives', nb_negatives)

        # First, we will calculate loss of the positive cases.
        # print(loss.shape, loss[positive_mask].shape)
        # TODO: take a look at this!!
        # positive_loss = tf.reduce_sum(loss[positive_mask], axis=[1, 2])
        positive_loss = tf.where(positive_mask, loss, 0.0)
        positive_loss = tf.reduce_mean(positive_loss, axis=[1, 2])
        tf.summary.scalar('positive loss', tf.reduce_mean(positive_loss))
        # tf.print('positive loss', positive_loss)
        # print('positive', positive_loss)

        # Base on the calculated loss function, we only choose the
        # hardest negative samples.
        negative_loss = tf.where(positive_mask, 0.0, loss)
        negative_loss = tf.reshape(negative_loss, shape=[-1, height * width])
        # print('batch_loss', batch_loss[batch_negative_mask])
        # tf.print('batch negative loss', negative_loss, negative_loss.shape)
        _, top_indices = tf.nn.top_k(negative_loss, k=nb_negatives)
        negative_loss = tf.gather(negative_loss, top_indices, axis=-1)
        # tf.print('negative loss', negative_loss)
        # print('negative', negative_loss)
        negative_loss = tf.reduce_mean(negative_loss, axis=[1, 2])
        tf.summary.scalar('negative loss', tf.reduce_sum(negative_loss))
        # tf.print('negative loss final', negative_loss)

        # Finally, return the final loss.
        return tf.reduce_mean(lampda * positive_loss + negative_loss)

    return hard_negative_mined_loss_fn

