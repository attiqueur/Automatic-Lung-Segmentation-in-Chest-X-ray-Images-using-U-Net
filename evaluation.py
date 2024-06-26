import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coefficient, intersection_over_union
from train import load_dataset, create_directory, create_dataset

IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_directory("Evaluation_Results")

    """ Loading model """
    with CustomObjectScope({'iou': intersection_over_union, 'dice_coefficient': dice_coefficient, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("files/model.h5")

    """ Dataset """
    dataset_path = "MontgomerySet"
    (train_x, train_y1, train_y2), (valid_x, valid_y1, valid_y2), (test_x, test_y1, test_y2) = load_dataset(dataset_path)

    """ Predicting the mask """
    for x, y1, y2 in tqdm(zip(test_x, test_y1, test_y2), total=len(test_x)):
        """ Extracting the image name """
        image_name = x.split("/")[-1]

        """ Reading the image """
        ori_x = cv2.imread(x, cv2.IMREAD_COLOR)
        ori_x = cv2.resize(ori_x, (IMAGE_WIDTH, IMAGE_HEIGHT))
        x = ori_x / 255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)

        """ Reading the mask """
        ori_y1 = cv2.imread(y1, cv2.IMREAD_GRAYSCALE)
        ori_y2 = cv2.imread(y2, cv2.IMREAD_GRAYSCALE)
        ori_y = ori_y1 + ori_y2
        ori_y = cv2.resize(ori_y, (IMAGE_WIDTH, IMAGE_HEIGHT))
        ori_y = np.expand_dims(ori_y, axis=-1)
        ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1)

        """ Predicting the mask """
        y_pred = model.predict(x)[0] > 0.5
        y_pred = y_pred.astype(np.int32)

        """ Saving the predicted mask along with the image and GT """
        save_image_path = f"Evaluation_Results/{image_name}"
        y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)

        sep_line = np.ones((IMAGE_HEIGHT, 10, 3)) * 255

        cat_image = np.concatenate([ori_x, sep_line, ori_y, sep_line, y_pred * 255], axis=1)
        cv2.imwrite(save_image_path, cat_image)
