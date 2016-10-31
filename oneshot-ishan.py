#!/usr/bin/python
import tensorflow as tf
import numpy as np

########################################
# Basic Architecture
# Controller network
#       ||||||
# Generate Key
#       ||||||
# Access Memory
#       ||||||
# Sigmoidal Output Layer


input_size = 1000
key_size = 1000
num_rows = 200
weight_decay = 0.8
num_classes = 1000

def weight_variable(shape):
    W = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(W)


def bias_variable(shape):
    b = tf.constant(0.1,shape=shape)
    return tf.Variable(b)


def memory_weights():
    W = tf.truncated_normal([key_size,1], stddev=0.1)
    return tf.Variable(W)


def linear_layer(input_vector,in_dim,out_dim):
    W = weight_variable([in_dim,out_dim])
    b = bias_variable([out_dim])
    return tf.nn.relu(tf.matmul(input_vector,W)+b)

def nsmall(a, n):
    k = n%num_rows
    return np.partition(a, k)[k-1]


#x = tf.placeholder(tf.float32,shape=[1,input_size])
#y_ = tf.placeholder(tf.float32,shape=[1,num_classes]);

x = tf.zeros([1,input_size])
y_ = tf.zeros([1,num_classes])

memory_module = tf.placeholder(tf.float32,[num_rows, key_size])

#################################
# Controller - 2 layer feed forward network
x1 = linear_layer(x,input_size,500)
key = linear_layer(x1,500,key_size)

key_norm = tf.nn.l2_normalize(key,1)
mat_norm = tf.nn.l2_normalize(memory_module,1)

similarity = tf.matmul(mat_norm, tf.transpose(key_norm))

read_weights = tf.nn.softmax(similarity)
data_read = tf.matmul(tf.transpose(read_weights),memory_module)

y = tf.nn.softmax(data_read)
error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))


######
# least used policy to be implemented
initial_read_weights = tf.zeros([num_rows,1])
previous_read_weights = tf.placeholder(tf.float32,shape=[num_rows,1])
least_used_weights = tf.placeholder(tf.float32,shape=[num_rows,1])

alpha = 1/(1+np.exp(-2))
write_weights = previous_read_weights*alpha + least_used_weights*(1-alpha)

initial_usage_weights = tf.zeros([num_rows,1])
previous_usage_weights = tf.placeholder(tf.float32,shape=[num_rows,1])
usage_weights = (weight_decay*previous_usage_weights) + (
                                                read_weights + write_weights)


memory_erase = tf.placeholder(tf.float32,shape=[num_rows,1])
initial_memory = tf.truncated_normal([num_rows, key_size],stddev=0.01)


memory = tf.add(tf.mul(memory_erase,memory_module),
                                        tf.matmul(write_weights,key))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
stored_usage_weights = sess.run(initial_usage_weights)
stored_read_weights = sess.run(initial_read_weights)
stored_memory = sess.run(initial_memory)
stored_least_used_weights = sess.run(initial_usage_weights)
memory_erase_vector = np.ones([num_rows,1])

for i in range(2000):
    stored_read_weights,stored_usage_weights,stored_memory = sess.run(
                                    [read_weights, usage_weights, memory],
                                    feed_dict=
                                    {memory_module: stored_memory,
                                    memory_erase: memory_erase_vector,
                                    least_used_weights: stored_least_used_weights,
                                    previous_usage_weights: stored_usage_weights,
                                    previous_read_weights: stored_read_weights})

    nthsmallest = nsmall(np.reshape(stored_usage_weights,[num_rows]),i+1)
    ind1 = stored_usage_weights > nthsmallest
    ind2 = stored_usage_weights <= nthsmallest
    stored_least_used_weights[ind1]=0
    stored_least_used_weights[ind2]=1
    memory_erase_vector = np.ones([num_rows,1])
    memory_erase_vector[np.argmin(stored_usage_weights)]=0
