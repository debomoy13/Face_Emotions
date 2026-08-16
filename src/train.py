import os
import matplotlib.pyplot as plt
from model import model

train_dir="data/split/train"
val_dir="data/split/val"
test_dir="data/split/test"

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    shuffle=False
)