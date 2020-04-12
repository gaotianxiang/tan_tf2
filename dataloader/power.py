import numpy as np
import wget
import zipfile
import os.path
import pickle
import os

import tensorflow as tf

tfd = tf.data


class Power:
    def __init__(self):
        self.path = './datasets/power/power.p'
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

        train = tfd.Dataset.from_tensor_slices(train).shuffle(100000).batch(batch_size, False).prefetch(
            tfd.experimental.AUTOTUNE)
        valid = tfd.Dataset.from_tensor_slices(valid).shuffle(100000).batch(batch_size, False).prefetch(
            tfd.experimental.AUTOTUNE)
        test = tfd.Dataset.from_tensor_slices(test).shuffle(100000).batch(batch_size, False).prefetch(
            tfd.experimental.AUTOTUNE)
        return train, valid, test


def load_data(path):
    """ Read uci dataloader files and return np array with
    gap, grp, voltage, gintens, sm, time """
    with open(path) as f:
        content = f.readlines()

    data = []
    for i, row in enumerate(content[1:]):
        fields = row.split(';')
        time = fields[1].split(':')
        time = 60 * float(time[0]) + float(time[1]) + np.random.rand()
        try:
            data.append(np.array([float(v) for v in fields[2:]] + [time]))
        except ValueError:
            pass

    data = np.array(data)
    return data


def load_data_split_with_noise(path):
    rng = np.random.RandomState(42)

    data = load_data(path)
    rng.shuffle(data)
    N = data.shape[0]

    data = np.delete(data, 3, axis=1)
    data = np.delete(data, 1, axis=1)
    ############################
    # Add noise
    ############################
    voltage_noise = 0.01 * rng.rand(N, 1)
    gap_noise = 0.001 * rng.rand(N, 1)
    sm_noise = rng.rand(N, 3)
    time_noise = np.zeros((N, 1))
    noise = np.hstack((gap_noise, voltage_noise, sm_noise, time_noise))
    data = data + noise

    N_test = int(0.1 * data.shape[0])
    data_test = data[-N_test:]
    data = data[0:-N_test]
    N_validate = int(0.1 * data.shape[0])
    data_validate = data[-N_validate:]
    data_train = data[0:-N_validate]

    return data_train, data_validate, data_test


def load_data_normalised(path):
    data_train, data_validate, data_test = load_data_split_with_noise(path)
    data = np.vstack((data_train, data_validate))
    mu = data.mean(axis=0)
    s = data.std(axis=0)
    data_train = (data_train - mu) / s
    data_validate = (data_validate - mu) / s
    data_test = (data_test - mu) / s

    return data_train, data_validate, data_test


def download_and_make_data(datapath):
    url = ('https://archive.ics.uci.edu/ml/machine-learning-databases/'
           '00235/household_power_consumption.zip')
    txtname = 'household_power_consumption.txt'
    path = os.path.join(datapath, 'power/')
    os.makedirs(path, exist_ok=True)
    # print('Downloading...')
    # filename = wget.download(url, path)
    filename = os.path.join(path, 'household_power_consumption.zip')
    print('\nExtracting...')
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall(path)
    zip_ref.close()
    print('Processing...')
    trn, val, tst = load_data_normalised(os.path.join(path, txtname))
    print('Saving...')
    outfile = os.path.join(path, 'power.p')
    pickle.dump(
        {'train': trn.astype(np.float32),
         'valid': val.astype(np.float32),
         'test': tst.astype(np.float32)},
        open(outfile, 'wb')
    )
