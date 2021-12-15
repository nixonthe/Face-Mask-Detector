import tensorflow as tf
from tensorflow.keras import layers


class CNNModel(tf.keras.Model):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')
        self.conv2 = layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')
        self.mp1 = layers.MaxPooling2D(pool_size=3)
        self.conv3 = layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')
        self.conv4 = layers.Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')
        self.mp2 = layers.MaxPooling2D(pool_size=3)
        self.gap = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.mp1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mp2(x)
        x = self.gap(x)
        x = self.dense1(x)
        out = self.dense2(x)
        return out
