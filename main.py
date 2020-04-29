import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk

import module.transforms as trans
from trainer import train
import dataloader
from module import StandardNegativeGaussianLogLikelihoods, MixtureLogLikelihoods


def main():
    # model = trans.Transformer([
    #     trans.LeakyReLU(),
    #     trans.LogRescale(), trans.RNNAdditiveCoupling(), trans.Reverse(), trans.LinearMap(), trans.LeakyReLU(),
    #     trans.LogRescale(), trans.RNNAdditiveCoupling(), trans.Reverse(), trans.LinearMap(), trans.LeakyReLU(),
    #     trans.LogRescale(), trans.RNNAdditiveCoupling(), trans.Reverse(), trans.LinearMap(), trans.LeakyReLU(),
    #     trans.LogRescale(), trans.RNNAdditiveCoupling(), trans.Reverse(), trans.LinearMap(), trans.LeakyReLU(),
    #     trans.LogRescale(), trans.RNNAdditiveCoupling(), trans.Reverse(), trans.LinearMap(), trans.LeakyReLU(),
    #     trans.LogRescale()
    # ])

    dim = 2
    mask = np.arange(dim) % 2
    mask = mask.astype(np.float32)
    hidden = [32, 32]
    model = trans.Transformer([trans.MaskAffineCoupling(hidden, 1 - mask), trans.MaskAffineCoupling(hidden, mask)])
    # model = trans.Transformer([
    #     trans.MaskAffineCoupling(hidden, mask),
    #     trans.MaskAffineCoupling(hidden, 1 - mask),
    #     trans.MaskAffineCoupling(hidden, mask),
    #     trans.MaskAffineCoupling(hidden, 1 - mask),
    #     trans.MaskAffineCoupling(hidden, mask),
    #     trans.MaskAffineCoupling(hidden, 1 - mask),
    #     trans.MaskAffineCoupling(hidden, mask),
    #     trans.MaskAffineCoupling(hidden, 1 - mask),
    #     trans.MaskAffineCoupling(hidden, mask),
    #     trans.MaskAffineCoupling(hidden, 1 - mask),
    #     trans.MaskAffineCoupling(hidden, mask),
    # ])
    dtst = Moon()
    train_dl, valid_dl, test_dl = dtst.get_dl(64)
    lr_schedule = tfk.optimizers.schedules.ExponentialDecay(0.001, decay_steps=5000, decay_rate=0.5)
    optim = tfk.optimizers.Adam(learning_rate=lr_schedule)
    num_epochs = 100
    loss_fn = StandardNegativeGaussianLogLikelihoods()
    # loss_fn = MixtureLogLikelihoods(40)
    train(model, loss_fn, optim, train_dl, valid_dl, test_dl, num_epochs, 200)


if __name__ == '__main__':
    main()
