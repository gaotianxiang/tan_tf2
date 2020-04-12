import tensorflow as tf
import tensorflow.keras as tfk
import numpy as np


class StandardNegativeGaussianLogLikelihoods(tfk.layers.Layer):
    def __init__(self):
        super(StandardNegativeGaussianLogLikelihoods, self).__init__()

    def build(self, input_shape):
        pass

    def call(self, x, **kwargs):
        res = -0.5 * np.log(2 * np.pi) - 0.5 * tf.square(x)
        res = tf.reduce_sum(res, -1)
        res = -tf.reduce_mean(res)
        return res


class ConditionalParameters(tfk.layers.Layer):
    def __init__(self, num_components):
        super(ConditionalParameters, self).__init__()
        self.num_components = num_components

    def build(self, input_shape):
        dim = 3 * self.num_components
        self.rnn = tfk.layers.GRU(units=dim, return_sequences=True)

    def call(self, x, **kwargs):
        # x = tf.expand_dims(x, -1)
        n, d = x.get_shape()
        x = tf.concat((-tf.ones((n, 1)), x), axis=-1)[:, :-1]
        x = tf.expand_dims(x, -1)
        params = self.rnn(x)
        return params


class MixtureLogLikelihoods(tfk.layers.Layer):
    def __init__(self, num_components):
        super(MixtureLogLikelihoods, self).__init__()
        self.num_components = num_components

    def build(self, input_shape):
        self.conditional = ConditionalParameters(self.num_components)

    def call(self, x, **kwargs):
        params = self.conditional(x)
        x = tf.expand_dims(x, -1)
        logits, means, logsigmas = tf.split(params, 3, -1)
        sigmas = tf.exp(logsigmas)

        log_norm_consts = -logsigmas - 0.5 * np.log(2 * np.pi)
        log_kernel = -0.5 * tf.square((x - means) / sigmas)

        log_exp_terms = log_kernel + log_norm_consts + logits
        log_likelihoods = tf.reduce_logsumexp(log_exp_terms, -1) - tf.reduce_logsumexp(logits, -1)
        nll = -tf.reduce_mean(tf.reduce_sum(log_likelihoods, -1))
        return nll
