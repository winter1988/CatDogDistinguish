# -*- coding: UTF-8 -*-
import numpy as np
import tensorflow as tf
from captcha.image import ImageCaptcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

from numpy.random import seed

import dataset

seed(10)
from tensorflow import set_random_seed
set_random_seed(20)
#mini_batch
batch_size= 32
#image width and height were set 64
img_size = 64
#
classes = ['dogs', 'cats']
num_class = len(classes)
validation_size = 0.2
num_channels = 3
train_path = 'training_data'
filter_size_conv1 = 3
num_filter_conv1 = 32

filter_size_conv2 = 3
num_filter_conv2 = 32

filter_size_conv3 = 3
num_filter_conv3 = 64

fc_layer_size = 1024
#读取数据
data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)

def create_weight(shape):

    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):

    return tf.Variable(tf.constant(0.05, shape=[size]))

def create_convolutional_layer(input, num_input_channels,
                               conv_filter_size, num_filters):
    weight = create_weight(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    biases = create_biases(num_filters)
    layer = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input=input, filter=weight,
                                                   strides=[1, 1, 1, 1], padding='SAME'), biases))
    layer = tf.nn.max_pool(layer, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')
    return layer

def create_flatten_layer(layer):

    layer_shape = layer.get_shape()
    num_feature = layer_shape[1:4].num_elements()

    layer = tf.reshape(layer, [-1, num_feature])

    return layer

def create_fc_layer(inp, num_input, num_outputs, use_relu=True):

    weight = create_weight(shape=[num_input, num_outputs])
    biases = create_biases(num_outputs)
    layer = tf.nn.bias_add(tf.matmul(inp, weight), biases)

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

def create_conv(x):
    layer_conv1 = create_convolutional_layer(input=x, num_input_channels=num_channels,
                                             conv_filter_size=filter_size_conv1, num_filters=num_filter_conv1)

    layer_conv2 = create_convolutional_layer(input=layer_conv1, num_input_channels=num_filter_conv1
                                             , conv_filter_size=num_filter_conv2, num_filters=num_filter_conv2)

    layer_conv3 = create_convolutional_layer(input=layer_conv2, num_input_channels=num_filter_conv2,
                                             conv_filter_size=num_filter_conv3, num_filters=num_filter_conv3)
    return layer_conv3



total_iterations = 0


if __name__ == "__main__":

    train = 0
    #cnn 四维数组
    x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, num_channels], name='x')
    y_true = tf.placeholder(tf.float32, shape=[None, num_class], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)



    layer_conv=create_conv(x)

    layer_flat = create_flatten_layer(layer_conv)

    layer_fc1 = create_fc_layer(layer_flat,
                                num_input=layer_flat.get_shape()[1:4].num_elements(),
                                num_outputs=fc_layer_size, use_relu=True)
    layer_fc2 = create_fc_layer(layer_fc1, num_input=fc_layer_size,
                                num_outputs=num_class, use_relu=False)

    y_pred = tf.nn.softmax(logits=layer_fc2, name='y_pred')
    y_pred_cls = tf.argmax(y_pred, dimension=1)

    num_iteration = 8000
    cross_entrcop = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)

    cost = tf.reduce_mean(cross_entrcop)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

    correct_prediction = tf.equal(y_pred_cls, y_true_cls)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()
    if (train == 0):
        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            for i in range(total_iterations,
                           total_iterations + num_iteration):
                x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
                x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)
                feed_dict_tr = {x: x_batch,
                                y_true: y_true_batch}
                feed_dict_val = {x: x_valid_batch,
                                 y_true: y_valid_batch}

                sess.run(optimizer, feed_dict=feed_dict_tr)

                if i % int(data.train.num_examples / batch_size) == 0:
                    val_loss = sess.run(cost, feed_dict=feed_dict_val)
                    epoch = int(i / int(data.train.num_examples / batch_size))


                    acc = sess.run(accuracy, feed_dict=feed_dict_tr)
                    val_acc = sess.run(accuracy, feed_dict=feed_dict_val)
                    msg = "Training Epoch {0}--- iterations: {1}--- Training Accuracy: {2:>6.1%}, Validation Accuracy: {3:>6.1%},  Validation Loss: {4:.3f}"
                    print(msg.format(epoch + 1, i, acc, val_acc, val_loss))
                    saver.save(sess, './dogs-cats-model/dog-cat.ckpt', global_step=i)



    elif (train == 1):
        pass