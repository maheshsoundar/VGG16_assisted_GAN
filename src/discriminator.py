import tensorflow as tf
tf.keras.utils.set_random_seed(42)  # sets seeds for base-python, numpy and tf
import numpy as np
np.random.seed(42)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Flatten,Conv2D

class Discriminator:
    def __init__(self):
        pass

    def build_model(self,image_shape,vgg16):
        model = Sequential()
        model.add(vgg16)

        model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=image_shape))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model