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
mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

image_size = 28
input_size = image_size*image_size
key_size = 40
num_rows = 100
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


def linear_layer(input_vector,in_dim,out_dim, l_name = ""):
	with tf.variable_scope("RNN"):
	    W = tf.get_variable(l_name + "weight", (in_dim, out_dim),
	        initializer = tf.random_normal_initializer(0.0, 2.0/(in_dim+out_dim)))
	    b = tf.get_variable(l_name + "bias", out_dim,
	        initializer = tf.constant_initializer(0.1))
	return tf.nn.relu(tf.matmul(input_vector,W)+b)


def only_linear_layer(input_vector,in_dim,out_dim, l_name = ""):
	with tf.variable_scope("RNN"):
	    W = tf.get_variable(l_name + "weight", (in_dim, out_dim),
	        initializer = tf.random_normal_initializer(0.0, 2.0/(in_dim+out_dim)))
	    b = tf.get_variable(l_name + "bias", out_dim,
	        initializer = tf.constant_initializer(0.1))
	return (tf.matmul(input_vector,W)+b)


def non_trainable_variable(shape, name = None, init = ['normal', 0.0, 0.1]):
    with tf.variable_scope("RNN"):
        if init[0] == "normal":
            ntv =  tf.get_variable(name, shape, trainable = False,
                initializer = tf.random_normal_initializer(init[1], init[2]))
        else:
            ntv =  tf.get_variable(name, shape,  trainable = False,
                initializer = tf.constant_initializer(init[1]))
    return ntv

def set_k_top_to_zero(x,k):
    values, indices = tf.nn.top_k(x,k)
    my_range = tf.expand_dims(tf.range(0, indices.get_shape()[0]), 1)
    my_range_repeated = tf.tile(my_range, [1, k])
    full_indices = tf.concat(2,[tf.expand_dims(my_range_repeated,2), tf.expand_dims(indices, 2)])
    full_indices = tf.reshape(full_indices, [-1, 2])
    to_substract = tf.sparse_to_dense(full_indices, x.get_shape(), tf.reshape(values, [-1]), default_value=0., validate_indices=False)
    res = (x - to_substract)
    return tf.sign(res)

seq_size = 200

x = tf.placeholder(tf.float32,shape=[None, seq_size, input_size + num_classes])
modified_x = tf.transpose(x,[1,0,2])
y_ = tf.placeholder(tf.float32,shape=[None, seq_size,num_classes]);


read_weights = non_trainable_variable([batch_size, num_rows, num_read_heads],\
										 name = "W_r", init = ['normal', 0.0, 0.1])
usage_weights = non_trainable_variable([batch_size,num_rows],\
										name = "W_u", init = ["constant", 1.0])
memory = non_trainable_variable([batch_size, num_rows, key_size], \
										name = "memory", init = ['normal', 0.0, 1e-4])
error = non_trainable_variable(1, name = "err", init = ["constant", 0.0])
accuracy = non_trainable_variable(1, name = "acc", init = ["constant", 0.0])
accuracy_next_100 = error = non_trainable_variable(1, name = "acc100", init = ["constant", 0.0])

#################################
#Controller
num_hidden = 200
initializer = tf.random_uniform_initializer(-1,1)
cell = rnn_cell.LSTMCell(num_hidden, initializer=initializer, state_is_tuple=True)

initial_state = cell.zero_state(batch_size, tf.float32)

state = initial_state
num_steps = seq_size
#inputs = [tf.squeeze(input_step, [1]) for input_step in tf.split(1, num_steps, x)]
#outputs, state = rnn.rnn(cell,inputs,dtype=tf.float32)


#acc_vector = []
with tf.variable_scope("RNN"):
    for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (output, state) = cell(tf.reshape(modified_x[time_step,:,:],shape=[batch_size,input_size+num_classes]), state)
	
        write_key_temp = linear_layer(output,num_hidden,num_read_heads*key_size, l_name = "write")
        write_key = tf.reshape(write_key_temp,shape=[-1,num_read_heads,key_size])

        alpha = linear_layer(output,num_hidden,num_read_heads, l_name = "interpolator")
        beta = tf.reshape(tf.sigmoid(alpha),shape=[-1,1,num_read_heads])

        least_used_weights = set_k_top_to_zero(usage_weights,num_rows-num_read_heads)

        write_weights = read_weights*beta + tf.reshape(least_used_weights,shape=[-1,num_rows,1])*(1-beta)

        memory1 = tf.sub(memory, tf.mul(memory,tf.reshape(least_used_weights,shape=[-1,num_rows,1])))
        memory = tf.add(memory1, tf.batch_matmul(write_weights,write_key))

        read_keys = linear_layer(output,num_hidden,num_read_heads*key_size, l_name = "poop_read")
        keys = tf.reshape(read_keys,shape=[-1,num_read_heads,key_size])
        key_norm = tf.nn.l2_normalize(keys,2)
        mat_norm = tf.nn.l2_normalize(memory,2)

        similarity = tf.batch_matmul(mat_norm, tf.transpose(key_norm,[0,2,1]))
        temp_similarity = tf.nn.softmax(tf.reshape(tf.transpose(similarity,[0,2,1]),shape=[-1,num_rows]))
        read_weights = tf.transpose(tf.reshape(temp_similarity,shape=[-1,num_read_heads,num_rows]),[0,2,1])

        data_read = tf.batch_matmul(tf.transpose(read_weights,[0,2,1]),memory)
        flattened_data_read = tf.reshape(data_read,shape=[-1,num_read_heads*key_size])

        usage_weights = (weight_decay*usage_weights) + num_read_heads*(tf.reshape(tf.reduce_mean(read_weights+write_weights,2),shape=[-1,num_rows]))
        state_h = state.h #linear_layer(outputs[time_step],num_hidden,num_hidden)

        y = (only_linear_layer(flattened_data_read,num_read_heads*key_size,num_classes, l_name = "read2out")
            + only_linear_layer(state_h,num_hidden,num_classes, l_name = "hidden2out"))

        if (time_step < 100):
            error += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, tf.reshape(y_[:,time_step,:],shape=[-1,num_classes])))

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(tf.reshape(y_[:,time_step,:],shape=[-1,num_classes]), 1))
        accuracy += tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        if (time_step > 99):
            accuracy_next_100 += tf.reduce_mean(tf.cast(correct_prediction, tf.float32))    #Zero out least used memory

#train_step = tf.train.RMSPropOptimizer(1e-4,decay=0.95,momentum=0.9).minimize(error)
train_step = tf.train.AdamOptimizer(1e-3).minimize(error)

sess = tf.Session()
saver = tf.train.Saver(max_to_keep=200)
shuffle_vector=range(num_classes)


sess.run(tf.initialize_all_variables())
avg_err = 0


num_episodes = 100000
for episode in range(num_episodes):
    if episode%500 == 0:
        save_path=saver.save(sess,"./model_"+str(episode)+".ckpt")
    random.shuffle(shuffle_vector)
    batch_xs, original_class = mnist.train.next_batch(batch_size*(seq_size+1))
    batch_ys = original_class[:,np.argsort(shuffle_vector)]
    batch_xs = np.reshape(batch_xs,[batch_size,seq_size+1,input_size])
    batch_ys = np.reshape(batch_ys,[batch_size,seq_size+1,num_classes])
    in_x = np.concatenate((batch_xs[:,1:],batch_ys[:,:seq_size]),axis=2)
    acc_next, acc, err, step = sess.run([accuracy_next_100, accuracy, error, train_step], feed_dict={x:in_x, y_:batch_ys[:,1:]})
    avg_err += err
    if episode%100 == 0:
        print "Episode: " + str(episode) + "Total accuracy " + str(acc) + " next accuracy " + str(acc_next) + " error " + str(err) + " avg error " + str(avg_err/(episode+1))


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
    acc_next, acc, err = sess.run([accuracy_next_100, accuracy, error], feed_dict={x:in_x, y_:batch_ys[:,1:]})
    print acc_next
#    print temp
    A += acc_next

print A/num_episodes


sess.close()
