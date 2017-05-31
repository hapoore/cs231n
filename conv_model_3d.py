import numpy as np
import tensorflow as tf

def decode_frames_avg_pool(feats):
    W1 = tf.get_variable("W1", shape=[feats.get_shape().as_list()[1], 30],
        initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", shape=[30], initializer=tf.constant_initializer(0.0))
    a = tf.matmul(feats, W1) + b1
    #y_pred = tf.nn.relu(a)
    y_pred = a
    # y_pred = tf.contrib.layers.fully_connected(inputs=avg, num_outputs=30, 
    #     activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(),
    #     biases_initializer=tf.constant_initializer(0.0), trainable=True)
    return y_pred, a

def encode_frames_3d_conv(X):
    (_, n_frames, height, width, n_channels) = X.get_shape().as_list()
    Wconv1 = tf.get_variable("Wconv1", shape=[3, 7, 7, 3, 32], initializer=tf.contrib.layers.xavier_initializer())
    bconv1 = tf.get_variable("bconv1", shape=[32], initializer=tf.constant_initializer(0.0))
    a1 = tf.nn.conv3d(X, Wconv1, strides=[1,1,2,2,1], padding='SAME') + bconv1
    h1 = tf.nn.relu(a1)
    p1 = tf.nn.max_pool3d(h1, ksize=[1,1,2,2,1], strides=[1,1,2,2,1], padding='SAME')
    Wconv2 = tf.get_variable("Wconv2", shape=[3, 5, 5, 32, 32], initializer=tf.contrib.layers.xavier_initializer())
    bconv2 = tf.get_variable("bconv2", shape=[32], initializer=tf.constant_initializer(0.0))
    a2 = tf.nn.conv3d(p1, Wconv2, strides=[1,1,2,2,1], padding='SAME') + bconv2
    h2 = tf.nn.relu(a2)
    p2 = tf.nn.max_pool3d(h2, ksize=[1,1,2,2,1], strides=[1,1,2,2,1], padding='SAME')
    (_, d_out, h_out, w_out, c_out) = p2.get_shape().as_list()
    reshaped_out = tf.reshape(p2, (-1, h_out*w_out*c_out*d_out))
    return reshaped_out


def simple_model(X):
    frames = encode_frames_3d_conv(X)
    y_pred, before_relu = decode_frames_avg_pool(frames)
    return y_pred, frames, before_relu

if __name__ == "__main__":
    X = tf.placeholder(tf.float32, [None, 10, 270, 270, 3])
    y = tf.placeholder(tf.int64, [None])
    y_pred = simple_model(X)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y)
    mean_loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(1e-3)

