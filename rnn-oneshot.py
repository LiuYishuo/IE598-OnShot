#!/usr/bin/python
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import random
########################################
# Basic Architecture
# Controller network
#       ||||||
# Generate Key
#       ||||||
# Access Memory
#       ||||||
# Sigmoidal Output Layer
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


image_size = 28
input_size = image_size*image_size
key_size = 40
num_rows = 128
weight_decay = 0.95
num_classes = 10
num_read_heads = 4

shuffle_vector = range(num_classes)

def weight_variable(shape):
    with tf.variable_scope("RNN"):
        W = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(W)


def bias_variable(shape):
    with tf.variable_scope("RNN"):
        b = tf.constant(0.1,shape=shape)
        return tf.Variable(b)


def linear_layer(input_vector,in_dim,out_dim):
    W = weight_variable([in_dim,out_dim])
    b = bias_variable([out_dim])
    return tf.nn.relu(tf.matmul(input_vector,W)+b)


def only_linear_layer(input_vector,in_dim,out_dim):
    W = weight_variable([in_dim,out_dim])
    b = bias_variable([out_dim])
    return (tf.matmul(input_vector,W)+b)

def non_trainable_variable(shape):
    with tf.variable_scope("RNN"):
        ntv = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(ntv, trainable=False)

def nsmall(a, n):
    k = n%num_rows
    return np.partition(a, k)[k-1]


seq_size = 5

x = tf.placeholder(tf.float32,shape=[seq_size, input_size + num_classes])
y_ = tf.placeholder(tf.float32,shape=[seq_size,num_classes]);

memory = non_trainable_variable([num_rows, key_size])
usage_weights = non_trainable_variable([num_rows])
read_weights = non_trainable_variable([ num_rows, num_read_heads])

error = tf.Variable(tf.constant(0.0),trainable=False)
accuracy = tf.Variable(tf.constant(0.0),trainable=False)

#################################
#Controller

num_hidden = 200
initializer = tf.random_uniform_initializer(-1,1)
cell = tf.nn.rnn_cell.LSTMCell(num_hidden, initializer=initializer)
initial_state = cell.zero_state(1, tf.float32)

state = initial_state
num_steps = seq_size


with tf.variable_scope("RNN"):
    for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (output, state) = cell(tf.reshape(x[time_step],shape=[1,input_size+num_classes]), state)

        write_key = linear_layer(output,num_hidden,key_size)
        alpha = linear_layer(output,num_hidden,1)
        beta = tf.sigmoid(alpha)
        least_used_weights = tf.sigmoid(1/usage_weights)
        write_weights = tf.reduce_mean(read_weights,1)*beta + least_used_weights*(1-beta)

        memory = tf.add(memory, tf.matmul(tf.transpose(write_weights),write_key))

        read_keys = linear_layer(output,num_hidden,num_read_heads*key_size)
        keys = tf.reshape(read_keys,shape=[num_read_heads,key_size])
        key_norm = tf.nn.l2_normalize(keys,1)
        mat_norm = tf.nn.l2_normalize(memory,1)
        similarity = tf.matmul(mat_norm, tf.transpose(key_norm))
        read_weights = tf.nn.softmax(similarity,dim=0)

        usage_weights = (weight_decay*usage_weights) + (tf.reshape(tf.reduce_mean(read_weights,1),shape=[1,num_rows]) +write_weights)

        data_read = tf.matmul(tf.transpose(read_weights),memory)
        flattened_data_read = tf.reshape(data_read,shape=[1,num_read_heads*key_size])

        y = (only_linear_layer(flattened_data_read,num_read_heads*key_size,num_classes))#+
#        only_linear_layer(state.h,num_hidden,num_classes))

        error = error + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_[time_step]))
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(tf.reshape(y_[time_step],shape=[1,num_classes]), 1))
        accuracy = accuracy + tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #Zero out least used memory
    #Compute lease used weights

train_step = tf.train.RMSPropOptimizer(1e-4,decay=0.95,momentum=0.9).minimize(error)



sess = tf.Session()
saver = tf.train.Saver(max_to_keep=200)
sess.run(tf.initialize_all_variables())


avg_acc = 0



num_episodes = 15000
for episode in range(num_episodes):
    if episode%500 == 0:
        save_path=saver.save(sess,"model_"+str(episode)+".ckpt")

    batch_xs, batch_ys = mnist.train.next_batch(seq_size+1)
    in_x = np.concatenate((batch_xs[1:],batch_ys[:seq_size]),axis=1)

    acc, err, step = sess.run([accuracy, error, train_step], feed_dict={x:in_x, y_:batch_ys[1:]})

    if episode%50 == 0:
        print "Episode :" + str(episode) + " accuracy " + str(acc) + " error " + str(err)


#saver.restore(sess,'./model_9500.ckpt')
A=0
num_episodes = 1000
for episode in range(num_episodes):
    batch_xs, batch_ys = mnist.test.next_batch(seq_size+1)
    in_x = np.concatenate((batch_xs[1:],batch_ys[:seq_size]),axis=1)

    acc, err = sess.run([accuracy, error], feed_dict={x:in_x, y_:batch_ys[1:]})
    A = A + acc

print A/num_episodes/seq_size

sess.close()
