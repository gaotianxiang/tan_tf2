import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import os
import argparse

import module.transforms as trans
from trainer import train, generate
import dataloader
from module import StandardNegativeGaussianLogLikelihoods, MixtureLogLikelihoods


def main():
    # define and parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--num', default=100, type=int)
    args = parser.parse_args()

    # tf config
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(device=gpu, enable=True)

    # use trans.Transformer() to stack transformers
    # model = trans.Transformer([
    #     # trans.LeakyReLU(),
    #     trans.LogRescale(), trans.RNNAdditiveCoupling(), trans.Reverse(), trans.LinearMap(), trans.LeakyReLU(),
    #     trans.LogRescale(), trans.RNNAdditiveCoupling(), trans.Reverse(), trans.LinearMap(), trans.LeakyReLU(),
    #     trans.LogRescale(), trans.RNNAdditiveCoupling(), trans.Reverse(), trans.LinearMap(), trans.LeakyReLU(),
    #     trans.LogRescale(), trans.RNNAdditiveCoupling(), trans.Reverse(), trans.LinearMap(), trans.LeakyReLU(),
    #     trans.LogRescale(), trans.RNNAdditiveCoupling(), trans.Reverse(), trans.LinearMap(), trans.LeakyReLU(),
    #     trans.LogRescale()
    # ])

    # model = trans.Transformer([
    #     trans.LinearMap(), trans.LeakyReLU(), trans.RNNAdditiveCoupling(), trans.Reverse(), trans.LogRescale(),
    #     trans.LinearMap(), trans.LeakyReLU(), trans.RNNAdditiveCoupling(), trans.Reverse(), trans.LogRescale(),
    #     trans.LinearMap(), trans.LeakyReLU(), trans.RNNAdditiveCoupling(), trans.Reverse(), trans.LogRescale(),
    #     trans.LinearMap(), trans.LeakyReLU(), trans.RNNAdditiveCoupling(), trans.Reverse(), trans.LogRescale(),
    #     trans.LinearMap(), trans.LeakyReLU(), trans.RNNAdditiveCoupling(), trans.Reverse(), trans.LogRescale(),
    # ])

    # hyper parameters
    batch_size = 256
    num_epochs = 100
    initial_lr = 0.005
    lr_decay_step = 5000
    lr_decay_rate = 0.5
    log_interval = 200
    ckpt_dir = './ckpts'
    summary_dir = './logs'

    # get data loaders
    dtst = dataloader.MNIST()
    train_dl, valid_dl, test_dl = dtst.get_dl(batch_size)

    # define flow model
    dim = np.product(dtst.dim)
    mask = np.arange(dim) % 2
    mask = mask.astype(np.float32)
    hidden = [32, 32]
    model = trans.Transformer([
        trans.MaskAffineCoupling(hidden, mask),
        trans.MaskAffineCoupling(hidden, 1 - mask),
        trans.MaskAffineCoupling(hidden, mask),
        trans.MaskAffineCoupling(hidden, 1 - mask),
        trans.MaskAffineCoupling(hidden, mask),
        trans.MaskAffineCoupling(hidden, 1 - mask),
        trans.MaskAffineCoupling(hidden, mask),
        trans.MaskAffineCoupling(hidden, 1 - mask),
        trans.MaskAffineCoupling(hidden, mask),
        trans.MaskAffineCoupling(hidden, 1 - mask),
        trans.MaskAffineCoupling(hidden, mask),
    ])

    # get optimizers and set up lr decay strategy
    lr_schedule = tfk.optimizers.schedules.ExponentialDecay(
        initial_lr, decay_steps=lr_decay_step, decay_rate=lr_decay_rate, staircase=True)
    optim = tfk.optimizers.Adam(learning_rate=lr_schedule)

    # choose loss function ram or std gaussian
    loss_fn = StandardNegativeGaussianLogLikelihoods()
    # loss_fn = MixtureLogLikelihoods(40)

    # define ckpt and ckpt manager
    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, os.path.join(ckpt_dir, repr(dtst)), max_to_keep=1)

    # define summary writer
    train_summary = tf.summary.create_file_writer(os.path.join(summary_dir, 'train'))
    val_summary = tf.summary.create_file_writer(os.path.join(summary_dir, 'val'))
    test_summary = tf.summary.create_file_writer(os.path.join(summary_dir, 'test'))

    # call function according to args.mode
    if args.mode is 'train':
        train(model, loss_fn, optim, train_dl, valid_dl,
              test_dl, num_epochs, train_summary, val_summary,
              test_summary, ckpt, ckpt_manager, log_interval=log_interval)
    elif args.mode is 'gen':
        generate(model, args.num, dtst.dim, test_summary, ckpt, ckpt_manager, restore=True)


if __name__ == '__main__':
    main()
