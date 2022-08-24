import cv2 as cv
from functools import partial
import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import Metric


def extract_bounding_boxes(y, threshold=0.5):
    if y.shape[-1] == 2:
        label_img = np.argmax(y, axis=-1)
    else:
        label_img = np.squeeze(np.where(y > threshold, 1, 0))

    label_img = np.asarray(label_img, dtype=np.uint8)
    contours, _ = cv.findContours(label_img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    boxes = [cv.boundingRect(c) for c in contours]

    return boxes


def bb_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union given between two bboxes.

    :param bbox1: tuple of (x, y, w, h) of the first bbox.
    :param bbox2: tuple of (x, y, w, h) of the second bbox.
    :returns: IoU.
    """
    bbox1_area = bbox1[2] * bbox1[3]
    bbox2_area = bbox2[2] * bbox2[3]

    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
    y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])

    if x1 >= x2 or y1 >= y2:
        intersection_area = 0
    else:
        intersection_area = (x2 - x1) * (y2 - y1)

    union_area = bbox1_area + bbox2_area - intersection_area
    return intersection_area / union_area


def bb_confusion_matrix(gt, pred, iou_threshold=0.5, pred_threshold=0.5):
    """
    Calculate confusion matrix for the given groundtruth and predicted value.

    :returns: a tuple of (true positive, true negatives, false positive, false negative)
    """
    gt_bboxes = extract_bounding_boxes(gt, pred_threshold)
    pred_bboxes = extract_bounding_boxes(pred, pred_threshold)

    tp, fn = 0, 0
    for gt_box in gt_bboxes:
        iou = [bb_iou(gt_box, box) for box in pred_bboxes]

        # If there is no remaining bounding boxes,
        # then the current gt box is considered as false negative.
        if not len(iou):
            fn += 1
            continue

        max_idx = max(range(len(iou)), key=lambda i: iou[i])

        # If the value of the max index is larger than the threshold,
        # we can consider this to be a true positive.
        # If that is the case, then, we increment the true positive count,
        # and remove that bounding box out of the list.
        if iou[max_idx] >= iou_threshold:
            tp += 1
            pred_bboxes.pop(max_idx)
        else:
            # Otherwise, there is no predicted bounding box that
            # matches the current groundtruth box.
            # In that case, we will increment the false negative count.
            fn += 1

    # After looping through all groundtruth boxes,
    # the remaining predicted boxes will be counted as false positive.
    fp = len(pred_bboxes)

    # True negative is always 0,
    # because we don't have any box in which there is no object.
    return tp, 0, fp, fn


class BBoxesIoUMetric(Metric):
    def __init__(self, iou_threshold=0.5, pred_threshold=0.5, name=None):
        super().__init__(name)
        self._iou_threshold = iou_threshold
        self._pred_threshold = pred_threshold

        self._tp = self.add_weight(name='tp', initializer='zeros', dtype=tf.int64)
        self._fp = self.add_weight(name='fp', initializer='zeros', dtype=tf.int64)
        self._fn = self.add_weight(name='fn', initializer='zeros', dtype=tf.int64)

    def update_state(self, y_true, y_pred, sample_weight=None):
        tp, fp, fn = tf.numpy_function(
                partial(BBoxesIoUMetric.iou_confusion_matrix,
                    iou_threshold=self._iou_threshold,
                    pred_threshold=self._pred_threshold),
                [y_true, y_pred],
                [tf.int64, tf.int64, tf.int64])

        self._tp.assign_add(tf.cast(tp, dtype=tf.int64))
        self._fp.assign_add(tf.cast(fp, dtype=tf.int64))
        self._fn.assign_add(tf.cast(fn, dtype=tf.int64))

    def result(self):
        return self._tp / (self._tp + self._fp + self._fn)

    def reset_states(self):
        for var in [self._tp, self._fp, self._fn]:
            var.assign(0)

    @classmethod
    def iou_confusion_matrix(cls, y_true, y_pred, iou_threshold, pred_threshold):
        tp, fp, fn = 0, 0, 0
        for gt, pred in zip(y_true, y_pred):
            a_tp, _, a_fp, a_fn = bb_confusion_matrix(gt, pred, iou_threshold, pred_threshold)
            tp += a_tp
            fp += a_fp
            fn += a_fn

        return tp, fp, fn


class ExtendedBBoxesIoUMetric(BBoxesIoUMetric):
    def __init__(self, ext_radius=20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ext_radius = ext_radius

    def update_state(self, y_true, y_pred, *args, **kwargs):
        pass
