import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import os
import pickle

tfd = tf.data


def plot_figure(data, path):
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1])
    plt.savefig(path)


def download_and_make_data(data_dir):
    data_dir = os.path.join(data_dir, 'moon')
    os.makedirs(data_dir, exist_ok=True)

    data, _ = make_moons(20000, shuffle=True, noise=0.05)
    train = data[:15000]
    valid = data[15000: 17500]
    test = data[17500:]

    plot_figure(train, os.path.join(data_dir, 'train.png'))
    plot_figure(valid, os.path.join(data_dir, 'valid.png'))
    plot_figure(test, os.path.join(data_dir, 'test.png'))

    save_dict = {
        'train': train,
        'valid': valid,
        'test': test
    }

    with open(os.path.join(data_dir, 'moon.p'), 'wb') as file:
        pickle.dump(save_dict, file)


class Moon:
    def __init__(self):
        self.path = './datasets/moon/moon.p'
        if not os.path.exists(self.path):
            download_and_make_data('./datasets')

    def get_dl(self, batch_size):
        with open(self.path, 'rb') as file:
            data = pickle.load(file)
        train = data['train']
        valid = data['valid']
        test = data['test']
        print(train.shape)
        print(valid.shape)
        print(test.shape)

        train = tfd.Dataset.from_tensor_slices(train).shuffle(10000).batch(batch_size, False).prefetch(
            tfd.experimental.AUTOTUNE)
        valid = tfd.Dataset.from_tensor_slices(valid).shuffle(10000).batch(batch_size, False).prefetch(
            tfd.experimental.AUTOTUNE)
        test = tfd.Dataset.from_tensor_slices(test).shuffle(10000).batch(batch_size, False).prefetch(
            tfd.experimental.AUTOTUNE)
        return train, valid, test


if __name__ == '__main__':
    download_and_make_data('../datasets')
