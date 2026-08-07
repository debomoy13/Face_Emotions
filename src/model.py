import torch
import torch.nn as nn
from PIL import glob
from tensorflow.keras.models import Sequential
classes=glob('data/')


filepath='data'
import tensorflow as tf

model=tf.keras.Sequential(
    [
        tf.keras.Input(48,48,1),

        tf.keras.layers.Conv2D(32,kernel_size=3,activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.Conv2D(64,kernel_size=3,activation='relu'),
        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.Conv2D(128, kernel_size=3,activation='relu'),
        tf.keras.layers.MaxPool2D(),

        tf.keras.layers.flatten,
        ])


    