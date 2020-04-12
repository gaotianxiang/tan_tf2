import tensorflow as tf
import tensorflow.keras as tfk
import datetime
import numpy as np

from module import transforms as trans
from module.conditional import StandardNegativeGaussianLogLikelihoods, MixtureLogLikelihoods
from dataloader import Power, Gas
from tqdm import tqdm, trange
import os


@tf.function
def train_step(model, loss_fn, optimizer, x_batch_data):
    with tf.GradientTape() as tape:
        transformed_x = model(x_batch_data, training=True)
        loss = loss_fn(transformed_x)
        loss += sum(model.losses)
    grads = tape.gradient(loss, model.trainable_weights)

    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss


@tf.function
def val_step(model, loss_fn, x_batch_data):
    transformed_x = model(x_batch_data, training=False)
    loss = loss_fn(transformed_x)
    loss += sum(model.losses)
    return loss


def val(model, loss_fn, data_loader):
    loss_mean = tfk.metrics.Mean()
    loss_mean.reset_states()

    for x_batch_val in data_loader:
        loss = val_step(model, loss_fn, x_batch_val)
        loss_mean.update_state(loss)
    return loss_mean.result()


def train(model, loss_fn, optimizer, train_data, valid_data, test_data, num_epochs, log_interval=500):
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/' + current_time + '/train'
    valid_log_dir = 'logs/' + current_time + '/valid'
    ckpt_dir = 'ckpts/' + current_time
    # train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    # test_summary_writer = tf.summary.create_file_writer(valid_log_dir)
    best_test_nll = float('inf')
    train_loss = tfk.metrics.Mean()
    for epoch in range(num_epochs):
        train_loss.reset_states()
        tqdm.write('start of epoch {}'.format(epoch))
        # with tqdm(total=len(train_data)) as progress_bar:
        for step, x_batch_train in enumerate(train_data):
            loss = train_step(model, loss_fn, optimizer, x_batch_train)
            # if los
            train_loss.update_state(loss)

            if step % log_interval == 0:
                # with train_summary_writer.as_default():
                #     tf.summary.scalar('loss', loss, step=optimizer.iterations)
                # progress_bar.update()
                print('epoch {} step {} loss {}'.format(epoch, step, train_loss.result()))
        # with train_summary_writer.as_default():
        #     tf.summary.scalar('loss_epoch', train_loss.result(), step=epoch)
        print('Epoch - {} NLL {:.5f}'.format(epoch, train_loss.result()))
        loss_val = val(model, loss_fn, test_data)
        # with test_summary_writer.as_default():
        #     tf.summary.scalar('loss_epoch', loss_val, step=epoch)
        print('\t Epoch - {} NLL {:.5f}'.format(epoch, loss_val))
        if loss_val < best_test_nll:
            model.save_weights(os.path.join(ckpt_dir, 'best.ckpt.pth'))
            best_test_nll = loss_val


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

    dim = 8
    mask = np.arange(dim) % 2
    mask = mask.astype(np.float32)
    hidden = [32, 32]
    # model = trans.Transformer([trans.MaskAffineCoupling(hidden, 1 - mask)])
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
    dtst = Gas()
    train_dl, valid_dl, test_dl = dtst.get_dl(1024)
    lr_schedule = tfk.optimizers.schedules.ExponentialDecay(0.01, decay_steps=5000, decay_rate=0.5)
    optim = tfk.optimizers.Adam(learning_rate=lr_schedule)
    num_epochs = 100
    loss_fn = StandardNegativeGaussianLogLikelihoods()
    # loss_fn = MixtureLogLikelihoods(40)
    train(model, loss_fn, optim, train_dl, valid_dl, test_dl, num_epochs)


if __name__ == '__main__':
    main()
