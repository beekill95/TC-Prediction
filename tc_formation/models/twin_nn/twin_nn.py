from . import blocks
import numpy as np
from sklearn import metrics
import tensorflow as tf
from tensorflow import keras


def normalize_to_dist_pred(pred: np.ndarray, weight: np.ndarray) -> np.ndarray:
    weight_len = np.linalg.norm(weight)
    return np.abs(pred) / weight_len


class TwinNN:
    def __init__(self, input_shape, fully_connected_hidden_layers, name=None):
        self._base_model = blocks.BaseBlock(input_shape, name=f'{name}_base')
        pos_model, pos_output_layer = blocks.FullyConnectedBlock(
            self._base_model,
            fully_connected_hidden_layers,
            name=f'{name}_pos_model')
        neg_model, neg_output_layer = blocks.FullyConnectedBlock(
            self._base_model,
            fully_connected_hidden_layers,
            name=f'{name}_neg_model')

        self._pos_model = pos_model
        self._pos_output_layer = pos_output_layer
        self._neg_model = neg_model
        self._neg_output_layer = neg_output_layer

        # Final model.
        self._model = keras.Model(
            inputs=self._base_model.inputs,
            outputs=dict(pos=self._pos_model.outputs[0],
                         neg=self._neg_model.outputs[0]),
            name=name,
        )

    def fit(self, *args, **kwargs):
        """Train the model by delegating the call to `keras.Model.fit`."""
        self._model.fit(*args, **kwargs)

    def predict_raw(self, *args, **kwargs):
        output = self._model.predict(*args, **kwargs)

        pos_pred = normalize_to_dist_pred(
            output['pos'],
            self._pos_output_layer.get_weights()[0])
        neg_pred = normalize_to_dist_pred(
            output['neg'],
            self._neg_output_layer.get_weights()[0])

        return dict(pos=pos_pred, neg=neg_pred)

    def predict(self, *args, **kwargs):
        output = self.predict_raw(*args, **kwargs)
        pos_pred = output['pos']
        neg_pred = output['neg']

        return np.where(pos_pred < neg_pred, 1, -1).flatten()

    def evaluate(self, ds: tf.data.Dataset) -> dict:
        pred = self.predict(ds)
        pred = np.where(pred == -1, 0, 1)
        print(pred, pred.shape)
        true = np.concatenate(list(ds.map(lambda _, y: y))).flatten()
        print(true.shape)
        true = np.where(true == -1, 0, 1)

        matched = true == pred
        diff = true != pred
        print(dict(
            true_positives=matched[true==1].sum(),
            true_negatives=matched[true==0].sum(),
            false_positives=diff[pred==1].sum(),
            false_negatives=diff[pred==0].sum(),
        ))

        return dict(
            precision=metrics.precision_score(true, pred),
            recall=metrics.recall_score(true, pred),
            f1_score=metrics.f1_score(true, pred),
        )

    def compile(self, *args, **kwargs):
        """Compile the model before training by delegating the call to `keras.Model.compile`"""
        self._model.compile(*args, **kwargs)

    def summary(self):
        return self._model.summary()
