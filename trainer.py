import tensorflow as tf
import tensorflow.keras as tfk


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


def train(model, loss_fn, optimizer, train_data, valid_data=None, test_data=None, num_epochs=100,
          train_summary_writer=None, valid_summary_writer=None, test_summary_writer=None,
          ckpt=None, ckpt_manager=None, log_interval=200):
    best_test_nll = float('inf')
    train_loss = tfk.metrics.Mean()
    for epoch in range(num_epochs):
        train_loss.reset_states()
        print('start of epoch {}'.format(epoch))
        for step, x_batch_train in enumerate(train_data):
            loss = train_step(model, loss_fn, optimizer, x_batch_train)
            train_loss.update_state(loss)
            if step % log_interval == 0:
                if train_summary_writer is not None:
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', loss, step=optimizer.iterations)
                print('\tepoch {} step {} loss {}'.format(epoch, step, loss))
        if train_summary_writer is not None:
            with train_summary_writer.as_default():
                tf.summary.scalar('loss_epoch', train_loss.result(), step=epoch)
        print('Epoch - {} NLL {:.5f}'.format(epoch, train_loss.result()))

        loss_val = val(model, loss_fn, test_data)
        if test_summary_writer is not None:
            with test_summary_writer.as_default():
                tf.summary.scalar('loss_epoch', loss_val, step=epoch)
        print('\t Epoch - {} NLL {:.5f}'.format(epoch, loss_val))
        if loss_val < best_test_nll:
            best_test_nll = loss_val
            if ckpt_manager is not None:
                ckpt_manager.save()
