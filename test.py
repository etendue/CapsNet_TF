import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.reset_default_graph()
graph = tf.Graph()
with graph.as_default():
    with tf.variable_scope("INPUT"):
        X = tf.placeholder(tf.float32, (None, 28, 28), name="X")
        y = tf.placeholder(tf.int32, (None, 1), name="y")
        y_onehot = tf.one_hot(y, 10, dtype=tf.float32)

    with tf.variable_scope("CONV1"):
        conv1 = tf.contrib.layers.conv2d(X, 256, kernel_size=9, stride=1, padding='VALID')
        conv1 = tf.identity(conv1, name="CONV1")

    with tf.variable_scope("PRIMARY_CAP"):
        conv2 = tf.contrib.layers.conv2d(conv1, 256, kernel_size=9, stride=2, padding='VALID')
        # [None, 6*6*256]
        conv2 = tf.identity(conv2, name="CONV2")
        u_I = tf.reshape(conv2, (-1, 9216))

    with tf.variable_scope("TRANSFORMATION"):
        # batch x 10 x 32x6x6x16 = batch x 10 x 1152 x 16 = 184320
        u_IJ = tf.contrib.layers.fully_connected(u_I, 184320)
        u_IJ = tf.reshape(u_IJ, [-1, 10, 1152, 16])
        u_IJ = tf.identity(u_IJ, name="u_IJ")

    with tf.variable_scope("ROUTING"):
        # 10x6x6x32 = 10x1152
        b_IJ = tf.Variable(np.zeros([10, 1152]), dtype=tf.float32, trainable=False)
        c_IJ = tf.nn.softmax(b_IJ)
        # 10x1152x1
        c_IJ = tf.expand_dims(c_IJ, axis=-1)
        # batch x 10x1152x16 -> batch x 10 x 16
        sj = tf.reduce_sum(tf.multiply(u_IJ, c_IJ), axis=-2)
        # do squash
        sj_mag2 = tf.reduce_sum(tf.square(sj), axis=-1, keep_dims=True)
        # batch x 10 x1
        prob = sj_mag2 / (1 + sj_mag2)
        # batch x 10 x1
        sj_mag = tf.sqrt(sj_mag2)
        # batch x 10 x 16
        sj_norm = sj / sj_mag
        # batch x 10 x 16 = batch x 10 x1 * batch x 10 x 16
        vj = prob * sj_norm
        # batch x 10 x 1 x16
        vj_expand = tf.expand_dims(vj, axis=2)
        # batch x 10 x 1152
        b_IJ = b_IJ + tf.reduce_sum(tf.reduce_sum(u_IJ * vj_expand, axis=-1), axis=0)

    with tf.variable_scope("COST"):
        los_true = tf.square(tf.maximum(0., 0.9 - sj_mag))
        los_false = tf.square(tf.maximum(0., sj_mag - 0.1))
        loss = y_onehot * los_true + 0.5 * (1 - y_onehot) * los_false
        margin_loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    with tf.variable_scope("TRAIN"):
        train_op = tf.train.AdamOptimizer().minimize(margin_loss)

    with tf.variable_scope("TEST"):
        out_prob = tf.squeeze(prob, axis=-1)
        predictions = tf.argmax(tf.nn.softmax(out_prob), axis=-1, output_type=tf.int32)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(y, predictions), tf.float32))


with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    #train_X,train_y = mnist.train.next_batch(0)