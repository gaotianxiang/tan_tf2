import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk

import module.transforms as trans
from trainer import train
import dataloader
from module import StandardNegativeGaussianLogLikelihoods, MixtureLogLikelihoods


def main():
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

    # define flow model
    dim = 63
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

    # get data loaders
    dtst = dataloader.BSDS300()
    train_dl, valid_dl, test_dl = dtst.get_dl(batch_size)

    # get optimizers and set up lr decay strategy
    lr_schedule = tfk.optimizers.schedules.ExponentialDecay(
        initial_lr, decay_steps=lr_decay_step, decay_rate=lr_decay_rate, staircase=True)
    optim = tfk.optimizers.Adam(learning_rate=lr_schedule)

    # choose loss function ram or std gaussian
    loss_fn = StandardNegativeGaussianLogLikelihoods()
    # loss_fn = MixtureLogLikelihoods(40)

    # call train function
    train(model, loss_fn, optim, train_dl, valid_dl,
          test_dl, num_epochs, log_interval=log_interval)


if __name__ == '__main__':
    main()
