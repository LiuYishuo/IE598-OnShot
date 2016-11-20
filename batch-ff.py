#!/usr/bin/python
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

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

batch_size = 16

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


seq_size = 200

x = tf.placeholder(tf.float32,shape=[None, seq_size, input_size + num_classes])

y_ = tf.placeholder(tf.float32,shape=[None, seq_size,num_classes]);

memory = non_trainable_variable([batch_size, num_rows, key_size])
read_weights = non_trainable_variable([batch_size, num_rows, num_read_heads])
usage_weights = tf.Variable(tf.constant(1.0,shape=[batch_size,num_rows]),trainable=False)
error = tf.Variable(tf.constant(0.0),trainable=False)
accuracy = tf.Variable(tf.constant(0.0),trainable=False)

#################################
#Controller
modified_x = tf.reshape(x,shape=[-1,input_size + num_classes])

num_hidden = 200
outputs = tf.transpose(tf.reshape(linear_layer(modified_x, input_size + num_classes, num_hidden),shape=[-1,seq_size,num_hidden]),[1,0,2])

with tf.variable_scope("RNN"):
    for time_step in range(seq_size):
        if time_step > 0: tf.get_variable_scope().reuse_variables()

        write_key = linear_layer(outputs[time_step],num_hidden,key_size)

        alpha = linear_layer(outputs[time_step],num_hidden,1)
        beta = tf.sigmoid(alpha)

        least_used_weights = tf.nn.softmax(tf.exp(1/usage_weights))
        write_weights = tf.reduce_mean(read_weights,2)*beta + least_used_weights*(1-beta)

        memory1 = tf.sub(memory, tf.mul(memory,tf.reshape(least_used_weights,shape=[-1,num_rows,1])))
        memory = tf.add(memory1, tf.matmul(tf.transpose(write_weights),write_key))

        read_keys = linear_layer(outputs[time_step],num_hidden,num_read_heads*key_size)
        keys = tf.reshape(read_keys,shape=[-1,num_read_heads,key_size])
        key_norm = tf.nn.l2_normalize(keys,2)
        mat_norm = tf.nn.l2_normalize(memory,2)

        similarity = tf.batch_matmul(mat_norm, tf.transpose(key_norm,[0,2,1]))
        temp_similarity = tf.nn.softmax(tf.reshape(tf.transpose(similarity,[0,2,1]),shape=[-1,num_rows]))
        read_weights = tf.transpose(tf.reshape(temp_similarity,shape=[-1,num_read_heads,num_rows]),[0,2,1])

        data_read = tf.batch_matmul(tf.transpose(read_weights,[0,2,1]),memory)
        flattened_data_read = tf.reshape(data_read,shape=[-1,num_read_heads*key_size])

        usage_weights = (weight_decay*usage_weights) + (tf.reshape(tf.reduce_mean(read_weights,2),shape=[-1,num_rows]) +write_weights)

        state_h = linear_layer(outputs[time_step],num_hidden,num_hidden)

        y = (only_linear_layer(flattened_data_read,num_read_heads*key_size,num_classes)+ only_linear_layer(state_h,num_hidden,num_classes))
        if (time_step < 100):
            error += time_step*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, tf.reshape(y_[:,time_step,:],shape=[-1,num_classes])))

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(tf.reshape(y_[:,time_step,:],shape=[-1,num_classes]), 1))
        accuracy += tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


train_step = tf.train.RMSPropOptimizer(1e-4,decay=0.95,momentum=0.9).minimize(error)

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=200)
shuffle_vector=range(num_classes)

sess.run(tf.initialize_all_variables())
avg_err = 0

num_episodes = 100000
for episode in range(num_episodes):
    if episode%500 == 0:
        save_path=saver.save(sess,"model_"+str(episode)+".ckpt")
    random.shuffle(shuffle_vector)
    batch_xs, original_class = mnist.train.next_batch(batch_size*(seq_size+1))
    batch_ys = original_class[:,np.argsort(shuffle_vector)]
    batch_xs = np.reshape(batch_xs,[batch_size,seq_size+1,input_size])
    batch_ys = np.reshape(batch_ys,[batch_size,seq_size+1,num_classes])
    in_x = np.concatenate((batch_xs[:,1:],batch_ys[:,:seq_size]),axis=2)
    acc, err, step = sess.run([accuracy, error, train_step], feed_dict={x:in_x, y_:batch_ys[:,1:]})
    avg_err += err
    if episode%10 == 0:
        print "Episode :" + str(episode) + " accuracy " + str(acc) + " error " + str(err) + " avg error " + str(avg_err/(episode+1))



#saver.restore(sess,'./model.ckpt')
A=0

num_episodes = 100

for episode in range(num_episodes):
    random.shuffle(shuffle_vector)
    batch_xs, original_class = mnist.test.next_batch((seq_size+1))
    batch_xs = np.reshape(np.repeat(batch_xs,batch_size),[-1,batch_size])
    batch_xs = np.reshape(np.transpose(batch_xs),[-1,input_size])
    batch_ys = original_class[:,np.argsort(shuffle_vector)]
    batch_ys = np.reshape(np.repeat(batch_ys,batch_size),[-1,batch_size])
    batch_ys = np.reshape(np.transpose(batch_ys),[-1,num_classes])
    batch_xs = np.reshape(batch_xs,[batch_size,seq_size+1,input_size])
    batch_ys = np.reshape(batch_ys,[batch_size,seq_size+1,num_classes])
    in_x = np.concatenate((batch_xs[:,1:],batch_ys[:,:seq_size]),axis=2)
    acc, err = sess.run([accuracy, error], feed_dict={x:in_x, y_:batch_ys[:,1:]})
    print acc
    A += acc

print A/num_episodes


sess.close()
