import tensorflow as tf
import tensorflow.keras as keras


class TwinNN(keras.Model):
    def __init__(
            self,
            *,
            base: keras.Model,
            pos_head: keras.Model,
            neg_head: keras.Model,
            pos_head_weight,
            neg_head_weight,
            **kwargs):
        super().__init__(**kwargs)

        self._base = base
        self._pos_head = pos_head
        self._pos_head_weight = pos_head_weight
        self._neg_head = neg_head
        self._neg_head_weight = neg_head_weight

        self._total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self._pos_loss_tracker = keras.metrics.Mean(name='pos_loss')
        self._neg_loss_tracker = keras.metrics.Mean(name='neg_loss')
        self._precision_tracker = keras.metrics.Precision(name='precision')
        self._recall_tracker = keras.metrics.Recall(name='recall')
        self._binary_accuracy_tracker = keras.metrics.BinaryAccuracy(name='binary_accuracy')

    def call(self, inputs, training=None):
        x = self._base(inputs, training=training)
        pos_out = self._pos_head(x, training=training)
        neg_out = self._neg_head(x, training=training)
        return self._output_labels(pos_out, neg_out)

    @property
    def metrics(self):
        return [
            self._total_loss_tracker,
            self._pos_loss_tracker,
            self._neg_loss_tracker,
            self._precision_tracker,
            self._recall_tracker,
            self._binary_accuracy_tracker,
        ]

    def train_step(self, inputs):
        X, y = inputs
        X = tf.cast(X, dtype=tf.float32)
        y_one_minus_one = tf.where(y == 0, -1, 1)

        with tf.GradientTape() as tape:
            x = self._base(X)
            pos_out = self._pos_head(x)
            neg_out = self._neg_head(x)

            # Calculate losses.
            total_loss, pos_loss, neg_loss = self._calc_losses(
                y_one_minus_one, pos_out, neg_out)

        weights = self.trainable_weights
        grads = tape.gradient(total_loss, weights)
        self.optimizer.apply_gradients(zip(grads, weights))

        # Update losses.
        self._total_loss_tracker.update_state(total_loss)
        self._pos_loss_tracker.update_state(pos_loss)
        self._neg_loss_tracker.update_state(neg_loss)

        # Update metrics.
        self._update_metrics(y, pos_out, neg_out)

        return self._calc_metrics()

    def test_step(self, inputs):
        X, y = inputs
        X = tf.cast(X, dtype=tf.float32)
        y_one_minus_one = tf.where(y == 0, -1, 1)

        x = self._base(X)
        pos_out = self._pos_head(x)
        neg_out = self._neg_head(x)

        # Calculate losses.
        total_loss, pos_loss, neg_loss = self._calc_losses(
            y_one_minus_one, pos_out, neg_out)

        # Update losses.
        self._total_loss_tracker.update_state(total_loss)
        self._pos_loss_tracker.update_state(pos_loss)
        self._neg_loss_tracker.update_state(neg_loss)

        # Update metrics.
        self._update_metrics(y, pos_out, neg_out)

        return self._calc_metrics()

    def _output_labels(self, pos_out, neg_out):
        pos_distance = tf.abs(pos_out) / tf.norm(self._pos_head_weight)
        neg_distance = tf.abs(neg_out) / tf.norm(self._neg_head_weight)
        return tf.where(pos_distance < neg_distance, 1., 0.)

    def _calc_losses(self, y_true, pos_out, neg_out):
        pos_mask = (y_true == 1)
        pos_loss = (outer_distance_loss(y_true[~pos_mask], pos_out[~pos_mask])
                    + inner_distance_loss(pos_out[pos_mask]))
        neg_loss = (outer_distance_loss(y_true[pos_mask], neg_out[pos_mask])
                    + inner_distance_loss(neg_out[~pos_mask]))
        total_loss = pos_loss + neg_loss

        return total_loss, pos_loss, neg_loss

    def _update_metrics(self, y_true, pos_out, neg_out):
        y_pred = self._output_labels(pos_out, neg_out)
        self._precision_tracker.update_state(y_true, y_pred)
        self._recall_tracker.update_state(y_true, y_pred)
        self._binary_accuracy_tracker.update_state(y_true, y_pred)

    def _calc_metrics(self):
        recall = self._recall_tracker.result()
        precision = self._precision_tracker.result()
        return {
            'loss': self._total_loss_tracker.result(),
            'pos_loss': self._pos_loss_tracker.result(),
            'neg_loss': self._neg_loss_tracker.result(),
            'binary_accuracy': self._binary_accuracy_tracker.result(),
            'precision': precision,
            'recall': recall,
            'f1': 2 * precision * recall / (precision + recall + 1e-6),
        }


def inner_distance_loss(y_pred: tf.Tensor):
    # return tf.reduce_mean(tf.square(y_pred))
    return tf.cond(tf.shape(y_pred)[0] == 0,
                   lambda: tf.constant(0.),
                   lambda: tf.reduce_mean(tf.square(y_pred)))


def outer_distance_loss(y_true, y_pred: tf.Tensor):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.tanh(y_pred)
    # return tf.reduce_mean(tf.square(y_true - y_pred))
    return tf.cond(tf.shape(y_pred)[0] == 0,
                   lambda: tf.constant(0.),
                   lambda: tf.reduce_mean(tf.square(y_true - y_pred)))
