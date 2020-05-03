import tensorflow as tf
import numpy as np

tfk = tf.keras


class Reverse(tfk.layers.Layer):
    def __init__(self):
        super(Reverse, self).__init__()

    def build(self, input_shape):
        pass

    def call(self, x, training=None, reverse=None):
        x = tf.reverse(x, axis=[-1])
        self.add_loss(0)
        return x


class Permutation(tfk.layers.Layer):
    def __init__(self, perm):
        super(Permutation, self).__init__()
        self.perm = perm

    def build(self, input_shape):
        self.inverse_perm = [0] * input_shape[-1]
        for i, p in enumerate(self.perm):
            self.inverse_perm[p] = i

    def call(self, x, training=None, reverse=False):
        if reverse:
            res = tf.gather(x, self.inverse_perm, axis=-1)
        else:
            res = tf.gather(x, self.perm, axis=-1)
        self.add_loss(0.0)
        return res


class LinearMap(tfk.layers.Layer):
    def __init__(self, trainable_A=True, trainable_b=True, irange=1e-10, init_A=None, init_b=None):
        super(LinearMap, self).__init__()
        self.trainable_A = trainable_A
        self.trainable_b = trainable_b
        self.irange = irange
        self.init_A = init_A
        self.init_b = init_b

    def build(self, input_shape):
        dim = input_shape[-1]
        self.weight = self.add_weight('weight', (dim, dim),
                                      initializer=tfk.initializers.Identity()
                                      if self.init_A is None else tfk.initializers.constant(self.init_A),
                                      trainable=self.trainable_A)
        self.bias = self.add_weight('bias', (dim,),
                                    initializer=tfk.initializers.zeros()
                                    if self.init_b is None else tfk.initializers.constant(self.init_b),
                                    trainable=self.trainable_b)

    def call(self, x, training=None, reverse=False):
        if reverse:
            weight_inv = tf.linalg.inv(self.weight)
            res = tf.linalg.matvec(weight_inv, x - self.bias)
            self.add_loss(-tf.math.log(tf.math.abs(tf.linalg.det(weight_inv)) + 1e-8))
        else:
            res = tf.linalg.matvec(self.weight, x) + self.bias
            self.add_loss(-tf.math.log(tf.math.abs(tf.linalg.det(self.weight)) + 1e-8))
        return res


class LeakyReLU(tfk.layers.Layer):
    def __init__(self, alpha=None):
        super(LeakyReLU, self).__init__()
        self.alpha = alpha

    def build(self, input_shape):
        self.alpha = self.add_weight('alpha_leaky_relu', shape=(1,),
                                     initializer=tfk.initializers.constant(
                                         self.alpha if self.alpha is not None else 5.0),
                                     trainable=True)

    def call(self, x, reverse=False):
        # n, _ = x.get_shape()
        alpha = tf.sigmoid(self.alpha)
        num_negative = tf.reduce_sum(tf.cast(tf.less(x, 0.0), tf.float32), -1)
        if reverse:
            res = tf.minimum(x, x / alpha)
            logdet = -num_negative * tf.math.log(alpha)
            self.add_loss(-tf.reduce_mean(logdet))
        else:
            res = tf.maximum(x, alpha * x)
            logdet = num_negative * tf.math.log(alpha)
            self.add_loss(-tf.reduce_mean(logdet))
        return res


class FullyConnect(tfk.layers.Layer):
    def __init__(self, hidden_sizes, irange=None, activation=tfk.activations.relu):
        super(FullyConnect, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.initializer = tfk.initializers.RandomUniform(-irange,
                                                          irange) if irange is not None else tfk.initializers.glorot_uniform
        self.activation = activation
        self.fc = tfk.Sequential()

    def build(self, input_shape):
        for dim in self.hidden_sizes:
            self.fc.add(tfk.layers.Dense(dim, activation=self.activation, kernel_initializer=self.initializer))

    def call(self, x, **kwargs):
        res = self.fc(x)
        return res


# class AdditiveCoupling(tfk.layers.Layer):
#     def __init__(self, hidden_sizes, irange=None, output_irange=None, activation=tfk.activations.relu):
#         super(AdditiveCoupling, self).__init__()
#         self.hidden_sizes = hidden_sizes
#         self.irange = irange
#         self.output_range = output_irange
#         self.activation = activation
#
#     def build(self, input_shape):
#         self.dim = input_shape[-1]
#         self.half_dim = input_shape[-1] // 2
#         self.fc = tfk.Sequential()
#         self.fc.add(FullyConnect(self.hidden_sizes, self.irange, self.activation))
#         self.fc.add(tfk.layers.Dense(self.dim - self.half_dim, activation=None,
#                                      kernel_initializer=None if self.output_range is None else
#                                      tfk.initializers.RandomUniform(-self.output_range, self.output_range)))
#
#     def call(self, x, training=True, reverse=False, **kwargs):
#         x1, x2 = tf.split(x, num_or_size_splits=[self.half_dim, self.dim - self.half_dim], axis=-1)
#         if reverse:
#             shift = self.fc(x1, training=training)
#             res = tf.concat((x1, x2 - shift), axis=-1)
#             self.add_loss(0.0)
#         else:
#             shift = self.fc(x1, training=training)
#             res = tf.concat((x1, x2 + shift), axis=-1)
#             self.add_loss(0.0)
#         return res


# class AffineCoupling(tfk.layers.Layer):
#     def __init__(self, hidden_sizes, activation=tfk.activations.relu):
#         super(AffineCoupling, self).__init__()
#         self.hidden_sizes = hidden_sizes
#         self.activation = activation
#
#     def build(self, input_shape):
#         self.dim = input_shape[-1]
#         self.half_dim = input_shape[-1] // 2
#         self.fc = tfk.Sequential([
#             FullyConnect(self.hidden_sizes, activation=self.activation),
#             tfk.layers.Dense(2 * (self.dim - self.half_dim))
#         ])
#
#     def call(self, x, reverse=False, **kwargs):
#         x1, x2 = tf.split(x, num_or_size_splits=[self.half_dim, self.dim - self.half_dim], axis=-1)
#         log_scale, shift = tf.split(self.fc(x1), num_or_size_splits=2, axis=-1)
#         if reverse:
#             x2 = (x2 - shift) / tf.exp(log_scale)
#             logdet = -tf.reduce_sum(log_scale, axis=-1)
#             self.add_loss(-tf.reduce_mean(logdet))
#             res = tf.concat((x1, x2), axis=-1)
#         else:
#             x2 = x2 * tf.exp(log_scale) + shift
#             logdet = tf.reduce_sum(log_scale, axis=-1)
#             self.add_loss(-tf.reduce_mean(logdet))
#             res = tf.concat((x1, x2), axis=-1)
#         return res


class LogRescale(tfk.layers.Layer):
    def __init__(self, init_zeros=True):
        super(LogRescale, self).__init__()
        self.init_zeros = init_zeros

    def build(self, input_shape):
        dim = input_shape[-1]
        self.log_scale = self.add_weight('log_scale', shape=(dim,),
                                         initializer=tfk.initializers.zeros() if self.init_zeros else None,
                                         trainable=True)

    def call(self, x, reverse=False, **kwargs):
        if reverse:
            res = x / tf.exp(self.log_scale)
            logdet = -tf.reduce_sum(self.log_scale)
            self.add_loss(-logdet)
        else:
            res = x * tf.exp(self.log_scale)
            logdet = tf.reduce_sum(self.log_scale)
            self.add_loss(-logdet)
        return res


class Shift(tfk.layers.Layer):
    def __init__(self, init_zeros=True):
        super(Shift, self).__init__()
        self.init_zeros = init_zeros

    def build(self, input_shape):
        dim = input_shape[-1]
        self.shift = self.add_weight('shift', shape=(dim,),
                                     initializer=tfk.initializers.zeros() if self.init_zeros else None,
                                     trainable=True)

    def call(self, x, reverse=False):
        if reverse:
            res = x - self.shift
        else:
            res = x + self.shift
        self.add_loss(0.0)
        return res


class Negative(tfk.layers.Layer):
    def __init__(self):
        super(Negative, self).__init__()

    def build(self, input_shape):
        pass

    def call(self, x, reverse=False):
        self.add_loss(0.0)
        return -x


class LogitTransform(tfk.layers.Layer):
    def __init__(self, alpha=0.05, max_val=256.0, logdet_mult=None):
        super(LogitTransform, self).__init__()
        self.alpha = alpha
        self.max_value = max_val
        self.logdet_mult = logdet_mult

    def build(self, input_shape):
        pass

    def call(self, x, reverse=False):
        if reverse:
            z = tf.sigmoid(x)
            res = (z - self.alpha) * self.max_value / (1 - self.alpha)
            logdet = tf.reduce_sum(x - 2 * tf.math.log(1 + tf.exp(x)) + tf.math.log(self.max_value)
                                   - tf.math.log(1 - self.alpha), axis=-1)
            self.add_loss(-tf.reduce_mean(logdet))
        else:
            sig = self.alpha + (1 - self.alpha) * x / self.max_value
            res = tf.math.log(sig) - tf.math.log(1 - sig)
            logdet = tf.reduce_sum(tf.math.log(1 - self.alpha) - tf.math.log(self.max_value) -
                                   tf.math.log(sig) - tf.math.log(1 - sig), axis=-1)
            self.add_loss(-tf.reduce_mean(logdet))
        return res


class RNNAdditiveCoupling(tfk.layers.Layer):
    def __init__(self, hidden_size):
        super(RNNAdditiveCoupling, self).__init__()
        self.hidden_size = hidden_size

    def build(self, input_shape):
        self.hidden_size = [hs for hs in self.hidden_size] + [1]
        self.rnn = tfk.layers.RNN([tfk.layers.GRUCell(units=hs) for hs in self.hidden_size],
                                  return_sequences=True, unroll=True)
        self.cells = self.rnn.cell

    def call(self, x, reverse=False):
        n, d = x.get_shape()
        if reverse:
            z = tf.expand_dims(x, -1)
            res = []
            inp = -tf.ones((n, 1))
            states = [tf.zeros((n, hs)) for hs in self.hidden_size]
            for i in range(d):
                output, hidden = self.cells(inp, states)
                inp = z[:, i] - output
                states = hidden
                res.append(inp)
            res = tf.concat(res, axis=1)
            self.add_loss(0.0)
        else:
            z = tf.concat((-tf.ones((n, 1)), x), axis=-1)
            z = tf.expand_dims(z, -1)[:, :-1]
            shift = self.rnn(z)
            shift = tf.squeeze(shift, axis=-1)
            res = x + shift
            self.add_loss(0.0)
        return res


class RNNAffineCoupling(tfk.layers.Layer):
    def __init__(self, hidden_size):
        super(RNNAffineCoupling, self).__init__()
        self.hidden_size = hidden_size

    def build(self, input_shape):
        self.hidden_size = [hs for hs in self.hidden_size] + [2]
        self.rnn = tfk.layers.RNN([tfk.layers.GRUCell(units=hs) for hs in self.hidden_size],
                                  return_sequences=True, unroll=True)
        self.cells = self.rnn.cell

    def call(self, x, reverse=False):
        n, d = x.get_shape()
        if reverse:
            z = tf.expand_dims(x, axis=-1)
            res = []
            inp = -tf.ones((n, 1))
            states = [tf.zeros((n, hs)) for hs in self.hidden_size]
            logdet = 0.0
            for i in range(d):
                output, hidden = self.cells(inp, states)
                log_scale, shift = tf.split(output, 2, axis=-1)
                inp = (z[:, i] - shift) / tf.exp(log_scale)
                logdet += -tf.reduce_sum(log_scale)
                states = hidden
                res.append(inp)
            self.add_loss(-tf.reduce_sum(logdet) / n)
            res = tf.concat(res, axis=1)
            res = tf.squeeze(res, axis=-1)
        else:
            z = tf.concat((-tf.ones((n, 1)), x), axis=-1)
            z = tf.expand_dims(z, axis=-1)[:, :-1]
            log_scale, shift = tf.split(self.rnn(z), num_or_size_splits=2, axis=-1)
            log_scale = tf.squeeze(log_scale, axis=-1)
            shift = tf.squeeze(shift, axis=-1)
            res = x * tf.exp(log_scale) + shift
            logdet = tf.reduce_sum(log_scale, -1)
            self.add_loss(-tf.reduce_mean(logdet))
        return res


class MaskAffineCoupling(tfk.layers.Layer):
    def __init__(self, hidden_sizes, mask=None):
        super(MaskAffineCoupling, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.mask = mask

    def build(self, input_shape):
        dim = input_shape[-1]
        self.mask = self.add_weight(name='mask', shape=(dim,),
                                    initializer=tfk.initializers.constant(self.mask if self.mask is not None
                                                                          else np.random.binomial(1, 0.5, (dim,))),
                                    trainable=False)
        self.fc = tfk.Sequential([
            FullyConnect(self.hidden_sizes),
            tfk.layers.Dense(dim * 2, activation=tfk.activations.tanh)
        ])

    def call(self, x, reverse=False, **kwargs):
        x_not_change = x * self.mask
        log_scale, shift = tf.split(self.fc(x_not_change), 2, -1)
        log_scale *= (1 - self.mask)
        shift *= (1 - self.mask)
        if reverse:
            res = (x - shift) / tf.exp(log_scale)
            logdet = -tf.reduce_sum(log_scale, -1)
            self.add_loss(-tf.reduce_mean(logdet))
        else:
            res = x * tf.exp(log_scale) + shift
            logdet = tf.reduce_sum(log_scale, -1)
            self.add_loss(-tf.reduce_mean(logdet))
        return res


class Transformer(tfk.Model):
    def __init__(self, transformations):
        super(Transformer, self).__init__()
        self.transformations = transformations

    def build(self, input_shape):
        pass

    def call(self, x, **kwargs):
        for transform in self.transformations:
            x = transform(x)
        return x

    def generate(self, x):
        for transform in reversed(self.transformations):
            x = transform(x, reverse=True, training=False)

        return x

