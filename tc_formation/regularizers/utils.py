import tensorflow as tf


def corr_coef(X):
    """
    Calculate correrlation matrix from a column matrix:
    each column is a sample.

    X should have shape (nb_features, nb_samples)
    """
    nb_samples = tf.shape(X)[1]
    feature_means = tf.reduce_mean(X, axis=1, keepdims=True)
    X_centered = X - feature_means

    # Calculate covariance matrix and variances of each feature.
    cov = X_centered @ tf.transpose(X_centered) / tf.cast(nb_samples, dtype=tf.float32)
    std = tf.sqrt(tf.linalg.diag_part(cov))

    # Calculate the correlation matrix.
    corr = cov / (std[None, ...] * std[..., None] + 1e-6)
    return corr


def cov(X):
    """
    Calculate covariance matrix from a column matrix:
    each column is a sample.

    X should have shape (nb_features, nb_samples)
    """
    nb_samples = tf.shape(X)[1]
    feature_means = tf.reduce_mean(X, axis=1, keepdims=True)
    X_centered = X - feature_means

    # Calculate covariance matrix and variances of each feature.
    cov_matrix = X_centered @ tf.transpose(X_centered) / tf.cast(nb_samples, dtype=tf.float32)
    return cov_matrix
