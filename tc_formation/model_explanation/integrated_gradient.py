import tensorflow as tf

def get_gradient(model, inputs, preprocessor=None):
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        if preprocessor:
            inputs = preprocessor(inputs)
        preds = model(inputs)

    return tape.gradient(preds, inputs)

def integrated_gradient(model, inp, baseline, preprocessor=None, nb_steps=127):
    """Calculate integrated gradient."""
    inputs = [baseline + (step / nb_steps) * (inp - baseline)
              for step in range(nb_steps + 1)]
    inputs = tf.stack(inputs)
    inputs = tf.cast(inputs, dtype=tf.float64)

    grads = get_gradient(model, inputs, preprocessor=preprocessor)
    grads = tf.where(tf.math.is_nan(grads), 0, grads)

    # Approximate integral.
    integral = (grads[1:] + grads[:-1]) / 2.0

    return tf.reduce_mean(integral, axis=0) * (inp - baseline)
