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
    y_pred = tf.contrib.layers.fully_connected(inputs=avg, num_outputs=30, 
        activation_fn=tf.nn.relu, weights_initializer=tf.contrib.layers.xavier_initializer(),
        biases_initializer=tf.constant_initializer(0.0), trainable=True)
    return y_pred

def resnet_avgpool(X):
    keras.layers.core.K.set_learning_phase(1)
    frames = encode_frames_resnet(X)
    y_pred = decode_frames_avg_pool(frames)
    return y_pred

def encode_frames_simple_conv(X):
    (_, n_frames, height, width, n_channels) = X.get_shape().as_list()
    flattened = tf.reshape(X, (-1, height, width, n_channels))
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    a1 = tf.nn.conv2d(flattened, Wconv1, strides=[1,2,2,1], padding='SAME') + bconv1
    h1 = tf.nn.relu(a1)
    Wconv2 = tf.get_variable("Wconv2", shape=[3, 3, 32, 32])
    bconv2 = tf.get_variable("bconv2", shape=[32])
    a2 = tf.nn.conv2d(h1, Wconv2, strides=[1,2,2,1], padding='SAME') + bconv2
    h2 = tf.nn.relu(a2)
    (_, h_out, w_out, c_out) = h2.get_shape().as_list()
    reshaped_out = tf.reshape(h2, (-1, n_frames, h_out*w_out*c_out))
    return reshaped_out


def simple_model(X):
    frames = encode_frames_simple_conv(X)
    y_pred = decode_frames_avg_pool(frames)
    return y_pred


if __name__ == "__main__":
    X = tf.placeholder(tf.float32, [None, 10, 270, 270, 3])
    y = tf.placeholder(tf.int64, [None])
    y_pred = simple_model(X)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y)
    mean_loss = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(1e-3)

