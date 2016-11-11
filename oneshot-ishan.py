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
    W = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(W)


def bias_variable(shape):
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


def nsmall(a, n):
    k = n%num_rows
    return np.partition(a, k)[k-1]

batch_size = 3
x = tf.placeholder(tf.float32,shape=[None, input_size])
prev_output = tf.placeholder(tf.float32,shape=[None, num_classes])
y_ = tf.placeholder(tf.float32,shape=[None, num_classes]);

memory_module = tf.placeholder(tf.float32,[None, num_rows, key_size])

#################################
output_size = num_read_heads*key_size + key_size + num_read_heads # one output for gate parameter

combined_input = tf.concat(1,[x, prev_output])
# Controller - 2 layer feed forward network
x1 = linear_layer(combined_input,input_size+num_classes,500)
x2 = linear_layer(x1,500,500)
x3 = linear_layer(x2,500,output_size)

read_keys = linear_layer(x2,500,num_read_heads*key_size)
write_key = tf.reshape(linear_layer(x2,500,key_size),shape=[-1,1,key_size])

alpha = linear_layer(x2,500,1)

keys = tf.reshape(read_keys,shape=[-1,num_read_heads,key_size])
key_norm = tf.nn.l2_normalize(keys,1)


######################################################################
# least used policy to be implemented
previous_read_weights = tf.placeholder(tf.float32,shape=[None,num_rows])
least_used_weights = tf.placeholder(tf.float32,shape=[None,num_rows])

beta = tf.sigmoid(alpha)
write_weights = tf.reshape(previous_read_weights*beta + least_used_weights*(1-beta),
                                            shape = [-1,num_rows,1])

previous_usage_weights = tf.placeholder(tf.float32,shape=[None,num_rows])

memory_erase = tf.placeholder(tf.float32,shape=[None,num_rows,1])
memory = tf.add(tf.mul(memory_erase,memory_module),
                                        tf.batch_matmul(write_weights,write_key))


mat_norm = tf.nn.l2_normalize(memory_module,2)
similarity = tf.batch_matmul(mat_norm, tf.transpose(key_norm,[0,2,1]))
#read_weights = tf.nn.softmax(similarity,dim=1)
read_weights = tf.transpose(tf.nn.softmax(tf.transpose(similarity,[0,2,1])),[0,2,1])

data_read = tf.batch_matmul(tf.transpose(read_weights,[0,2,1]),memory)

flattened_data_read = tf.reshape(data_read,shape=[-1,num_read_heads*key_size])
flattened_hidden_state = tf.reshape(x2,shape=[-1,500])

#input_to_softmax = tf.squeeze(tf.concat(1,[flattened_data_read, flattened_hidden_state]))
#y = only_linear_layer(input_to_softmax,num_read_heads*key_size+500,num_classes)
y = (only_linear_layer(flattened_data_read,num_read_heads*key_size,num_classes)+
            only_linear_layer(flattened_hidden_state,500,num_classes))

usage_weights = (weight_decay*previous_usage_weights) + (
                                                tf.reduce_mean(read_weights,2) +
                                                tf.reshape(write_weights,shape=[-1,num_rows]))


error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
train_step = tf.train.RMSPropOptimizer(1e-4,decay=0.95,momentum=0.9).minimize(error)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess = tf.InteractiveSession()
saver = tf.train.Saver(max_to_keep=200)
sess.run(tf.initialize_all_variables())

initial_usage_weights = tf.truncated_normal([batch_size,num_rows],stddev=0.1)
stored_usage_weights = sess.run(initial_usage_weights)

initial_read_weights = tf.truncated_normal([batch_size,num_rows],stddev=0.1)
stored_read_weights = sess.run(initial_read_weights)

initial_memory = tf.truncated_normal([batch_size,num_rows, key_size],stddev=0.2)
stored_memory = sess.run(initial_memory)

stored_least_used_weights = sess.run(initial_usage_weights)
memory_erase_vector = np.ones([batch_size,num_rows,1])


batch_xs1, batch_ys1 = mnist.train.next_batch(batch_size)
output = np.zeros([batch_size,num_classes])
avg_acc = 0


num_episodes = 100000
for episode in range(num_episodes):
    random.shuffle(shuffle_vector)
    if episode%500 == 0:
        save_path=saver.save(sess,"model_"+str(episode)+".ckpt")

    if episode%100 == 0:
        print "Episode " + str(episode) + " : "

    stored_usage_weights = np.zeros([batch_size,num_rows])
    stored_read_weights_flattened = np.zeros([batch_size,num_rows])
    stored_memory = np.zeros([batch_size,num_rows,key_size])
    stored_least_used_weights = np.zeros([batch_size,num_rows])
    memory_erase_vector = np.ones([batch_size,num_rows,1])

    for i in range(80):
        batch_xs, original_class = mnist.train.next_batch(batch_size)
        batch_ys = original_class[:,np.argsort(shuffle_vector)]
        #batch_ys = original_class
        prediction, stored_read_weights,stored_usage_weights,stored_memory, step, acc = sess.run(
                [y, read_weights, usage_weights, memory, train_step, error],
                                        feed_dict=
                                        {x: batch_xs, y_: batch_ys,
                                        prev_output: output,
                                        memory_module: stored_memory,
                                        memory_erase: memory_erase_vector,
                                        least_used_weights: stored_least_used_weights,
                                        previous_usage_weights: stored_usage_weights,
                                        previous_read_weights: stored_read_weights_flattened})

        output = batch_ys

        stored_read_weights_flattened = np.reshape(np.sum(stored_read_weights,2),[batch_size,num_rows])/num_read_heads

        stored_least_used_weights = np.zeros([batch_size,num_rows])
        memory_erase_vector = np.ones([batch_size,num_rows,1])
        for k in range(batch_size):
            nthsmallest_weight = nsmall(stored_usage_weights[k,:],num_read_heads+1)
            ind = stored_usage_weights[k,:] <= nthsmallest_weight
            stored_least_used_weights[k,ind] = 1
            memory_erase_vector[np.argmin(stored_usage_weights[k,:])]=0

    if episode%100 == 0:
        print acc

#print avg_acc/num_episodes

#saver.restore(sess,'./model-40500-.ckpt')

stored_usage_weights = np.zeros([1,num_rows])
stored_read_weights_flattened = np.zeros([1,num_rows])
stored_memory = np.zeros([1,num_rows,key_size])
stored_least_used_weights = np.zeros([1,num_rows])
memory_erase_vector = np.ones([1,num_rows,1])

random.shuffle(shuffle_vector)

output = np.zeros([1,num_classes])
total_acc = 0
for i in range(500):


    batch_xs, original_class = mnist.train.next_batch(1)
    batch_ys = original_class[:,np.argsort(shuffle_vector)]
    #batch_ys = original_class
    prediction, stored_read_weights,stored_usage_weights,stored_memory, acc = sess.run(
            [y, read_weights, usage_weights, memory, accuracy],
                                    feed_dict=
                                    {x: batch_xs, y_: batch_ys,
                                    prev_output: output,
                                    memory_module: stored_memory,
                                    memory_erase: memory_erase_vector,
                                    least_used_weights: stored_least_used_weights,
                                    previous_usage_weights: stored_usage_weights,
                                    previous_read_weights: stored_read_weights_flattened})


    output = batch_ys
    total_acc = total_acc + acc
    stored_read_weights_flattened = np.reshape(np.sum(stored_read_weights,2),[1,num_rows])/num_read_heads

    stored_least_used_weights = np.zeros([1,num_rows])
    memory_erase_vector = np.ones([1,num_rows,1])
    for k in range(1):
        nthsmallest_weight = nsmall(stored_usage_weights[k,:],num_read_heads+1)
        ind = stored_usage_weights[k,:] <= nthsmallest_weight
        stored_least_used_weights[k,ind] = 1
        memory_erase_vector[np.argmin(stored_usage_weights[k,:])]=0


print "Testing accuracy :"
print total_acc

sess.close()
