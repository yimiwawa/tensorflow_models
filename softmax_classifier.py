import time
import math
import numpy as np
import tensorflow as tf
from utils import data_iterator


def softmax(x):
    x_max = tf.expand_dims(tf.reduce_max(x, 1), 1)  # [1,2] => [[1],[2]]
    x = x - x_max
    x_exp = tf.exp(x)
    out = x_exp / tf.expand_dims(tf.reduce_sum(x_exp, 1), 1)
    return out


def cross_entropy_loss(y, yhat):
    y = tf.to_float(y)
    yhat_log = tf.log(yhat)
    out = 0.0 - tf.reduce_sum(y * yhat_log)
    return out

class Config(object):
    batch_size = 64
    n_samples = 1024
    n_features = 100
    n_classes = 5
    max_epochs = 50
    lr = 1e-4


class SoftmaxModel(object):
    def __init__(self, config):
        self.config = config
        self.load_data()
        self.add_placeholders()
        self.pred = self.add_model(self.input_placeholder)
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)

    def load_data(self):
        np.random.seed(1234)
        self.input_data = np.random.rand(
            self.config.n_samples, self.config.n_features)
        self.input_labels = np.ones((self.config.n_samples,), dtype=np.int32)

    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.float32,
                                                shape=(self.config.batch_size, self.config.n_features))
        self.labels_placeholder = tf.placeholder(tf.float32,
                                                 shape=(self.config.batch_size, self.config.n_classes))

    def create_feed_dict(self, input_batch, label_batch):
        if label_batch is None:
            feed_dict = {self.input_placeholder: input_batch}
        else:
            feed_dict = {
                self.input_placeholder: input_batch,
                self.labels_placeholder: label_batch
            }
        return feed_dict

    def add_training_op(self, loss):
        train_op = tf.train.GradientDescentOptimizer(self.config.lr) \
            .minimize(loss)
        return train_op

    def add_model(self, input_data):
        with tf.variable_scope("softmax"):
            W = tf.get_variable("weights", (self.config.n_features, self.config.n_classes),
                                initializer=tf.zeros_initializer)
            b = tf.get_variable("bias", (self.config.n_classes,), initializer=tf.zeros_initializer)
            out = softmax(tf.matmul(input_data, W) + b)
        return out

    def add_loss_op(self, pred):
        loss = cross_entropy_loss(self.labels_placeholder, pred)
        return loss

    def run_epoch(self, sess, input_data, input_labels):
        average_loss = 0
        for step, (input_batch, label_batch) in enumerate(
                data_iterator(input_data, input_labels,
                              batch_size=self.config.batch_size,
                              label_size=self.config.n_classes)):
            feed_dict = self.create_feed_dict(input_batch, label_batch)

            _, loss_value = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
            average_loss += loss_value

        average_loss = average_loss / step
        return average_loss

    def fit(self, sess, input_data, input_labels):
        losses = []
        for epoch in range(self.config.max_epochs):
            start_time = time.time()
            average_loss = self.run_epoch(sess, input_data, input_labels)
            duration = time.time() - start_time
            print('Epoch %d: loss = %.2f (%.3f sec)'
                  % (epoch, average_loss, duration))
            losses.append(average_loss)
        return losses


def test_SoftmaxModel():
    config = Config()
    with tf.Graph().as_default():
        model = SoftmaxModel(config)

        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)
        losses = model.fit(sess, model.input_data, model.input_labels)

    assert losses[-1] < .5
    print "Basic (non-exhaustive) classifier tests pass\n"


if __name__ == "__main__":
    test_SoftmaxModel()
