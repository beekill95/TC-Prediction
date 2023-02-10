import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers


class VAE(keras.Model):
    def __init__(self, encoder: keras.Model, decoder: keras.Model, **kwargs) -> None:
        super().__init__(**kwargs)
        self._encoder = encoder
        self._sampling_layer = SamplingLayer()
        self._decoder = decoder

        self._total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self._kl_loss_tracker = keras.metrics.Mean(name='kl_loss')
        self._reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')

    @property
    def metrics(self):
        return [
            self._total_loss_tracker,
            self._kl_loss_tracker,
            self._reconstruction_loss_tracker,
        ]

    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            inputs = tf.cast(inputs, dtype=tf.float32)
            z_mean, z_log_var = self._encoder(inputs)
            z = self._sampling_layer((z_mean, z_log_var))
            reconstruction = self._decoder(z)

            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum((inputs - reconstruction)**2, axis=(1, 2))
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = kl_loss + reconstruction_loss

        weights = self.trainable_weights
        grads = tape.gradient(total_loss, weights)
        self.optimizer.apply_gradients(zip(grads, weights))

        self._total_loss_tracker.update_state(total_loss)
        self._kl_loss_tracker.update_state(kl_loss)
        self._reconstruction_loss_tracker.update_state(reconstruction_loss)
        return {
            'loss': self._total_loss_tracker.result(),
            'kl_loss': self._kl_loss_tracker.result(),
            'reconstruction_loss': self._reconstruction_loss_tracker.result(),
        }


class SamplingLayer(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch_size = tf.shape(z_mean)[0]
        dim_size = tf.shape(z_mean)[1]
        epsilon = keras.backend.random_normal((batch_size, dim_size))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
