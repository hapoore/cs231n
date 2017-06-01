import numpy as np
import tensorflow as tf
import keras


""" Uses Pretrained ResNet on each frame, 
    returns a tensor with shape (batch_size, num_frames, 2048)
"""
def encode_frames_resnet(X):
    (_, n_frames, height, width, n_channels) = X.get_shape().as_list()
    flattened = tf.reshape(X, (-1, height, width, n_channels))
    resnet = keras.applications.ResNet50(include_top=False, 
        weights='imagenet', input_tensor=flattened, input_shape=(270, 270, 3), pooling=None)
    out_chans = resnet.output.get_shape().as_list()[3]
    reshaped_out = tf.reshape(resnet.output, (-1, n_frames, out_chans))
    return reshaped_out

""" Simple decoder that just does average pool across all frames
    and then uses fully connected layer to get predictions
"""
def decode_frames_avg_pool(frames):
    avg = tf.reduce_mean(frames, axis=1)
    W1 = tf.get_variable("W1", shape=[avg.get_shape().as_list()[1], 30],
        initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", shape=[30], initializer=tf.constant_initializer(0.0))
    a = tf.matmul(avg, W1) + b1
    #y_pred = tf.nn.relu(a)
    y_pred = a
    # y_pred = tf.contrib.layers.fully_connected(inputs=avg, num_outputs=30, 
    #     activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(),
    #     biases_initializer=tf.constant_initializer(0.0), trainable=True)
    return y_pred, a

def decode_frames_max_pool(frames):
    avg = tf.reduce_max(frames, axis=1)
    W1 = tf.get_variable("W1", shape=[avg.get_shape().as_list()[1], 30],
        initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", shape=[30], initializer=tf.constant_initializer(0.0))
    y_pred = tf.matmul(avg, W1) + b1
    return y_pred

def resnet_avgpool(X):
    keras.layers.core.K.set_learning_phase(1)
    frames = encode_frames_resnet(X)
    y_pred = decode_frames_avg_pool(frames)
    return y_pred

def encode_frames_simple_conv(X):
    (_, n_frames, height, width, n_channels) = X.get_shape().as_list()
    flattened = tf.reshape(X, (-1, height, width, n_channels))
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32], initializer=tf.contrib.layers.xavier_initializer())
    bconv1 = tf.get_variable("bconv1", shape=[32], initializer=tf.constant_initializer(0.0))
    a1 = tf.nn.conv2d(flattened, Wconv1, strides=[1,2,2,1], padding='SAME') + bconv1
    h1 = tf.nn.relu(a1)
    batchnorm1 = tf.contrib.layers.batch_norm(h1, .9, True, True, 1e-8)
    Wconv2 = tf.get_variable("Wconv2", shape=[3, 3, 32, 32], initializer=tf.contrib.layers.xavier_initializer())
    bconv2 = tf.get_variable("bconv2", shape=[32], initializer=tf.constant_initializer(0.0))
    a2 = tf.nn.conv2d(batchnorm1, Wconv2, strides=[1,2,2,1], padding='SAME') + bconv2
    h2 = tf.nn.relu(a2)
    batchnorm2 = tf.contrib.layers.batch_norm(h2, .9, True, True, 1e-8)
    (_, h_out, w_out, c_out) = batchnorm2.get_shape().as_list()
    reshaped_out = tf.reshape(batchnorm2, (-1, n_frames, h_out*w_out*c_out))
    return reshaped_out

def conv_norm(X, in_chans, out_chans, filter_size, stride, scope, is_training):
    with tf.variable_scope(scope):
        Wconv = tf.get_variable(scope + "Wconv", shape=[filter_size, filter_size, in_chans, out_chans],
                               initializer=tf.contrib.layers.xavier_initializer())
        bconv = tf.get_variable(scope + "bconv", shape=[out_chans],
                               initializer=tf.constant_initializer(0.))
        # conv - spatial batchnorm - relu
        a1 = tf.nn.conv2d(X, Wconv, strides=[1,stride,stride,1], padding='SAME') + bconv
        normed = tf.contrib.layers.batch_norm(a1, center=True, scale=True, 
                                              is_training=is_training, scope='bn',
                                              trainable=True)
        return normed

def res_block(X, in_chans, out_chans, scope, is_training):
    with tf.variable_scope(scope):
        dims = X.get_shape()[2]
        new_dims = int(dims) / 2
        out1 = conv_norm(X, in_chans, in_chans, 3, 1, "conv1", is_training)
        out1 = tf.nn.relu(out1)
        if in_chans == out_chans:
            out2 = conv_norm(out1, in_chans, out_chans, 3, 1, "conv2", is_training)
            return tf.nn.relu(out2 + X)
        else:
            out2 = conv_norm(out1, in_chans, out_chans, 3, 2, "conv2", is_training)
            Wshort = tf.get_variable(scope + "Wshort", shape=[1, 1, in_chans, out_chans],
                               initializer=tf.contrib.layers.xavier_initializer())
            shortcut = tf.nn.conv2d(X, Wshort, strides=[1,2,2,1], padding='SAME')
            return tf.nn.relu(out2 + shortcut)
    

def encode_frames_simple_resnet(X, is_training):
    # define our weights (e.g. init_two_layer_convnet)
    (_, n_frames, height, width, n_channels) = X.get_shape().as_list()
    flattened = tf.reshape(X, (-1, height, width, n_channels))
    out = conv_norm(flattened, 3, 32, 7, 2, "conv1", is_training)
    out = tf.nn.relu(out)
    n = 3
    for i in range(n-1):
        out = res_block(out, 32, 32, "res32_" + str(i), is_training)
    out = res_block(out, 32, 64, "res32_final", is_training)
    for i in range(n-1):
        out = res_block(out, 64, 64, "res16_" + str(i), is_training)
    out = res_block(out, 64, 128, "res16_final", is_training)   
    for i in range(n-1):
        out = res_block(out, 128, 128, "res8_" + str(i), is_training)
    out = res_block(out, 128, 128, "res8_final", is_training)
    y_out = tf.reshape(out, (-1, n_frames, 128*28*28))
    return y_out

def simple_model(X):
    frames = encode_frames_simple_conv(X)
    y_pred, before_relu = decode_frames_avg_pool(frames)
    return y_pred, frames, before_relu

def resnet_max_pool(X, is_training):
    frames = encode_frames_simple_resnet(X, is_training)
    y_pred = decode_frames_max_pool(frames)
    return y_pred


if __name__ == "__main__":
    X = tf.placeholder(tf.float32, [None, 10, 224, 224, 3])
    y = tf.placeholder(tf.int64, [None])
    y_pred = simple_model(X)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y)
    mean_loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(1e-3)

