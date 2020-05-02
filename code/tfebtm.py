import math

import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model


def batch_iter(biterms,
                embeddings,
                vocab_size,
                batch_size=4096,
                shuffle=True):
    '''
    return batches for unsupervised model
    biterms: list of tuple [(w1,w2)], w1 and w2 are indexes.
    embeddings: matrix of word embeddings, dtype=numpy.ndarray
    '''
    size = embeddings.shape
    data_len = len(biterms)
    num_batch = int((data_len - 1) / batch_size) + 1
    copy = biterms.copy()
    if shuffle:
        random.shuffle(copy)

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        batch_length = end_id - start_id
        idx_batch = copy[start_id:end_id]
        emb_batch = np.zeros((batch_length, size[1]))
        for j in range(len(idx_batch)):
            w1, w2 = idx_batch[j]
            emb_batch[j] = embeddings[w1] + embeddings[w2]
        yield idx_batch, emb_batch


@tf.function
def selected_production(x, theta):
    return tf.map_fn(lambda x: theta[x[0]] * theta[x[1]], x, tf.float32, 100, swap_memory=True)

class TFEBTM(Model):
    def __init__(self,
                 num_topics,
                 vocab_size,
                 t_hidden_size,
                 embeddings,
                 batch_size=4096,
                 theta_act="tanh",
                 enc_drop=0.5):
        super(TFEBTM, self).__init__()
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.t_hidden_size = t_hidden_size
        self.enc_drop = enc_drop
        self.batch_size = batch_size
        self.theta_act = theta_act
        self.training = True

        num_embeddings, emb_size = embeddings.shape
        self.emb_size = emb_size

        self.rho = tf.constant(embeddings)
        self.alphas = Dense(num_topics, use_bias=False)

        self.q_theta1 = Dense(t_hidden_size, activation=theta_act)
        self.q_theta2 = Dense(t_hidden_size, activation=theta_act)

        self.mu_q_theta = Dense(num_topics)
        self.logsigma_q_theta = Dense(num_topics)

        self.t_drop = tf.keras.layers.Dropout(enc_drop)

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def gaussian(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
            mu,logvar; dtype = float32
        """
        if self.training:
            std = tf.exp(0.5 * logvar)
            eps = tf.random.normal(std.shape)
            return eps * std + mu
        else:
            return mu

    def encode(self, biterms):
        """Returns paramters of the variational distribution for \\theta.

        input: biterms vectors with size batch_size * emb_size
        output: mu_theta, log_sigma_theta, kld_theta
        """
        q_theta1 = self.q_theta1(biterms)
        q_theta = self.q_theta2(q_theta1)
        if self.enc_drop > 0 and self.training:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        kld_theta = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + logsigma_theta - tf.pow(mu_theta, 2) -
                          tf.exp(logsigma_theta),
                          axis=-1))
        return mu_theta, logsigma_theta, kld_theta

    def get_beta(self):
        '''return the probability of P(w|z)'''
        logit = self.alphas(self.rho)
        beta = tf.transpose(tf.nn.softmax(
            logit, axis=0))  # softmax over vocab dimension
        return beta

    def get_theta(self, biterms):
        '''return topic distribution \\theat P(z|b)'''
        mu_theta, logsigma_theta, kld_theta = self.encode(biterms)
        ## mu, sigma of Gaussian, kld_theaa
        z = self.gaussian(mu_theta, logsigma_theta)
        theta = tf.nn.softmax(z, axis=-1)
        return theta, kld_theta

    def decode(self, bi_idx, theta, transposed_beta):
        temp = selected_production(bi_idx, transposed_beta)
        res = tf.reduce_sum((tf.pow(theta, 2) * temp), axis=1)
        preds = tf.math.log(res + 1e-6)
        return preds

    def call(self, bi_idx, biterms, theta=None):
        ## get \\theta
        if theta is None:
            theta, kld_theta = self.get_theta(biterms)
        else:
            kld_theta = None

        ## get \\beta
        beta = self.get_beta()

        ## get prediction loss
        preds = self.decode(bi_idx, theta, tf.transpose(beta))
        recon_loss = tf.reduce_mean(preds)
        return recon_loss, kld_theta
