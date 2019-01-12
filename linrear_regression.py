import tensorflow as tf
import numpy as np

X_data = np.arange(100, step=.1)
y_data = X_data + 20*np.sin(X_data/10)

n_samples = 1000
batch_size = 100
learningrate = 0.1

X_data = np.reshape(X_data, (n_samples, 1))
y_data = np.reshape(y_data, (n_samples, 1))

X = tf.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.placeholder(tf.float32, shape=(batch_size, 1))

with tf.variable_scope("linear-regression"):
    W = tf.get_variable("weights", (1, 1),
                        initializer=tf.random_normal_initializer())
    b = tf.get_variable("bias", (1,),
                        initializer=tf.constant_initializer(0.0))
    y_pred = tf.matmul(X, W) + b

loss = tf.reduce_sum((y - y_pred)**2/n_samples)
opt = tf.train.AdamOptimizer(learningrate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(500):
        indices = np.random.choice(n_samples, batch_size)
        X_batch, y_batch = X_data[indices], y_data[indices]

        _, loss_val = sess.run([opt, loss], feed_dict={X:X_batch, y:y_batch})
        print "iter:%d, loss:%.2f" % (i, loss_val)