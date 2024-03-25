import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K

def intersection_over_union(y_true, y_pred):
    """Calculates Intersection over Union (IoU) metric for segmentation.

    Args:
        y_true (tensor): Ground truth segmentation mask.
        y_pred (tensor): Predicted segmentation mask.

    Returns:
        float: Intersection over Union (IoU) score.
    """
    def calculate_iou(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        iou_score = (intersection + 1e-15) / (union + 1e-15)
        iou_score = iou_score.astype(np.float32)
        return iou_score
    return tf.numpy_function(calculate_iou, [y_true, y_pred], tf.float32)

smooth = 1e-15

def dice_coefficient(y_true, y_pred):
    """Calculates Dice Coefficient for segmentation.

    Args:
        y_true (tensor): Ground truth segmentation mask.
        y_pred (tensor): Predicted segmentation mask.

    Returns:
        float: Dice Coefficient score.
    """
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    dice_score = (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
    return dice_score

def dice_loss(y_true, y_pred):
    """Calculates Dice Loss for segmentation.

    Args:
        y_true (tensor): Ground truth segmentation mask.
        y_pred (tensor): Predicted segmentation mask.

    Returns:
        float: Dice Loss.
    """
    return 1.0 - dice_coefficient(y_true, y_pred)
