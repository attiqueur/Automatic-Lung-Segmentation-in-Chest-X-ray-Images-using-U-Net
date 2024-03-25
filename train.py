import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from model import build_unet
from metrics import dice_loss, dice_coefficient, intersection_over_union

""" Global parameters """
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512

def create_directory(path):
    """ Create a directory if it does not exist. """
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path, split=0.1):
    """ Load dataset from specified path and split into train, validation, and test sets. """
    images = sorted(glob(os.path.join(path, "CXR_png", "*.png")))
    masks_left = sorted(glob(os.path.join(path, "ManualMask", "leftMask", "*.png")))
    masks_right = sorted(glob(os.path.join(path, "ManualMask", "rightMask", "*.png")))

    split_size = int(len(images) * split)

    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
    train_y1, valid_y1 = train_test_split(masks_left, test_size=split_size, random_state=42)
    train_y2, valid_y2 = train_test_split(masks_right, test_size=split_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y1, test_y1 = train_test_split(train_y1, test_size=split_size, random_state=42)
    train_y2, test_y2 = train_test_split(train_y2, test_size=split_size, random_state=42)

    return (train_x, train_y1, train_y2), (valid_x, valid_y1, valid_y2), (test_x, test_y1, test_y2)

def read_image(path):
    """ Read and preprocess image from specified path. """
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (IMAGE_WIDTH, IMAGE_HEIGHT))
    x = x / 255.0
    x = x.astype(np.float32)
    return x

def read_mask(left_path, right_path):
    """ Read and preprocess mask from specified paths. """
    left_mask = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
    right_mask = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
    mask = left_mask + right_mask
    mask = cv2.resize(mask, (IMAGE_WIDTH, IMAGE_HEIGHT))
    mask = mask / np.max(mask)
    mask = mask > 0.5
    mask = mask.astype(np.float32)
    mask = np.expand_dims(mask, axis=-1)
    return mask

def parse_function(x, y1, y2):
    """ Parse function for TensorFlow dataset pipeline. """
    def parse_image(x, y1, y2):
        x = x.decode()
        y1 = y1.decode()
        y2 = y2.decode()

        x = read_image(x)
        y = read_mask(y1, y2)
        return x, y

    x, y = tf.numpy_function(parse_image, [x, y1, y2], [tf.float32, tf.float32])
    x.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    y.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    return x, y

def create_dataset(X, Y1, Y2, batch_size=8):
    """ Create TensorFlow dataset from input data. """
    dataset = tf.data.Dataset.from_tensor_slices((X, Y1, Y2))
    dataset = dataset.shuffle(buffer_size=200)
    dataset = dataset.map(parse_function)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(4)
    return dataset

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_directory("files")

    """ Hyperparameters """
    BATCH_SIZE = 2
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 10
    MODEL_PATH = os.path.join("files", "model.h5")
    CSV_PATH = os.path.join("files", "Data.csv")

    """ Dataset """
    DATASET_PATH = "MontgomerySet"
    (train_x, train_y1, train_y2), (valid_x, valid_y1, valid_y2), (test_x, test_y1, test_y2) = load_dataset(DATASET_PATH)

    print(f"Train: {len(train_x)} - {len(train_y1)} - {len(train_y2)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y1)} - {len(valid_y2)}")
    print(f"Test: {len(test_x)} - {len(test_y1)} - {len(test_y2)}")

    train_dataset = create_dataset(train_x, train_y1, train_y2, batch_size=BATCH_SIZE)
    valid_dataset = create_dataset(valid_x, valid_y1, valid_y2, batch_size=BATCH_SIZE)

    """ Model """
    model = build_unet((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    metrics = [dice_coefficient, intersection_over_union, Recall(), Precision()]
    model.compile(loss=dice_loss, optimizer=Adam(LEARNING_RATE), metrics=metrics)

    callbacks = [
        ModelCheckpoint(MODEL_PATH, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(CSV_PATH)
    ]

    model.fit(
        train_dataset,
        epochs=NUM_EPOCHS,
        validation_data=valid_dataset,
        callbacks=callbacks
    )
