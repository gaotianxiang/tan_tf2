import tensorflow as tf
import numpy as np

tfd = tf.data
tfk = tf.keras


class MNIST:
    def __init__(self):
        pass

    def get_dl(self, batch_size):
        (x_train, y_train), (x_test, y_test) = tfk.datasets.mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
        x_test = x_test.reshape(x_test.shape[0], -1).astype(np.float32)

        x_train /= 127.5
        x_train -= 1
        x_test /= 127.5
        x_test -= 1

        train = tfd.Dataset.from_tensor_slices(x_train).shuffle(10000).batch(batch_size).prefetch(
            tfd.experimental.AUTOTUNE)
        test = tfd.Dataset.from_tensor_slices(x_test).batch(batch_size).prefetch(tfd.experimental.AUTOTUNE)

        return train, None, test
