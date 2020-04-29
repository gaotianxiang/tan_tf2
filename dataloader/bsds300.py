import numpy as np
import imageio
import pickle
import wget
import glob
import os
import tarfile
import tensorflow as tf

tfd = tf.data


class BSDS300:
    def __init__(self):
        self.path = './datasets/bsds/bsds.p'
        if not os.path.exists(self.path):
            download_and_make_data('./datasets')

    def get_dl(self, batch_size):
        with open(self.path, 'rb') as file:
            data = pickle.load(file)
        train = data['train']
        valid = data['valid']
        test = data['test']
        train = tfd.Dataset.from_tensor_slices(train).shuffle(100000).batch(batch_size, False).prefetch(
            tfd.experimental.AUTOTUNE)
        valid = tfd.Dataset.from_tensor_slices(valid).shuffle(100000).batch(batch_size, False).prefetch(
            tfd.experimental.AUTOTUNE)
        test = tfd.Dataset.from_tensor_slices(test).shuffle(100000).batch(batch_size, False).prefetch(
            tfd.experimental.AUTOTUNE)
        return train, valid, test


def extract_patches(impath, step=2):
    """ Get grayscale image patches. """
    img = imageio.imread(impath, as_gray=True).astype(np.float32)
    h, w = img.shape
    patches = []
    for i in range(0, h - 7, step):
        for j in range(0, w - 7, step):
            patch = np.reshape(img[i:i + 8, j:j + 8], (64,)) + np.random.rand(64)
            patches.append(patch)
    return np.array(patches)


def process_images(imdir, extension='.jpg'):
    """ Extract all patches from images in a directory. """
    impaths = glob.glob(os.path.join(imdir, '*' + extension))
    im_patches = [extract_patches(ip) for ip in impaths]
    return np.concatenate(im_patches, 0)


def make_dataset(bsds_imdir, save_path):
    # Load patches, rescale, demean, drop last pixel
    print('Loading training data...')
    train_patches = process_images(os.path.join(bsds_imdir, 'train')) / 256.0
    train_patches = train_patches - np.mean(train_patches, 1, keepdims=True)
    train_patches = train_patches[:, :-1].astype(np.float32)
    print('Loading testing data...')
    test_patches = process_images(os.path.join(bsds_imdir, 'test')) / 256.0
    test_patches = test_patches - np.mean(test_patches, 1, keepdims=True)
    test_patches = test_patches[:, :-1].astype(np.float32)
    # Split.
    N_train = train_patches.shape[0]
    train_perm = np.random.permutation(N_train)
    N_valid = int(0.1 * N_train)
    # Save.
    pickle.dump(
        {'train': train_patches[train_perm[N_valid:]],
         'valid': train_patches[train_perm[:N_valid]],
         'test': test_patches},
        open(save_path, 'wb'))
    return train_patches, test_patches


def download_and_make_data(datapath):
    url = ('https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/'
           'BSDS300-images.tgz')
    path = os.path.join(datapath, 'bsds/')
    os.makedirs(path, exist_ok=True)
    print('Downloading...')
    filename = wget.download(url, path)
    print('\nExtracting {}...'.format(filename))
    tar = tarfile.open(filename, 'r')
    tar.extractall(path)
    tar.close()
    print('Processing and saving...')
    imgpath = os.path.join(path, 'BSDS300', 'images')
    savepath = os.path.join(path, 'bsds.p')
    make_dataset(imgpath, savepath)


if __name__ == "__main__":
    download_and_make_data('../datasets')
