#!/usr/bin/python
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

import numpy as np
import numpy.lib.index_tricks as ndi
import matplotlib.pyplot as plt
from scipy import ndimage
import random
import os

from mann.utils.generators import OmniglotGenerator
########################################
# Basic Architecture
# Controller network
#       ||||||
# Generate Key
#       ||||||
# Access Memory
#       ||||||
# Sigmoidal Output Layer

image_size = 20
input_size = image_size*image_size
key_size = 40
num_rows = 100
weight_decay = 0.95
num_classes = 5
num_read_heads = 4

batch_size = 16
seq_size = 49
shuffle_vector = range(num_classes)

# Helper functions for declaring weights and biases for any layer
def weight_variable(shape):
    W = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(W)


def bias_variable(shape):
    b = tf.constant(0.1,shape=shape)
    return tf.Variable(b)


# Single Hidden Linear Layer With RELU activation
# in_dim -> out_dim
# Each layer has a unique name, which is used to get the weights for that layer
def linear_layer(input_vector,in_dim,out_dim, l_name = ""):
	with tf.variable_scope("RNN"):
	    W = tf.get_variable(l_name + "weight", (in_dim, out_dim),
	        initializer = tf.random_normal_initializer(0.0, 2.0/(in_dim+out_dim)))
	    b = tf.get_variable(l_name + "bias", out_dim,
	        initializer = tf.constant_initializer(0.1))
	return tf.nn.relu(tf.matmul(input_vector,W)+b)


# Single Layer without activation
def only_linear_layer(input_vector,in_dim,out_dim, l_name = ""):
	with tf.variable_scope("RNN"):
	    W = tf.get_variable(l_name + "weight", (in_dim, out_dim),
	        initializer = tf.random_normal_initializer(0.0, 2.0/(in_dim+out_dim)))
	    b = tf.get_variable(l_name + "bias", out_dim,
	        initializer = tf.constant_initializer(0.1))
	return (tf.matmul(input_vector,W)+b)


# Create a variable that will not be modified by the back-prop
def non_trainable_variable(shape, name = None, init = ['normal', 0.0, 0.1]):
    with tf.variable_scope("RNN"):
        if init[0] == "normal":
            ntv =  tf.get_variable(name, shape, trainable = False,
                initializer = tf.random_normal_initializer(init[1], init[2]))
        else:
            ntv =  tf.get_variable(name, shape,  trainable = False,
                initializer = tf.constant_initializer(init[1]))
    return ntv


# Function to threshold a tensor
# Sets the k highest elements to 0, rest to 1
def set_k_top_to_zero(x,k):
    values, indices = tf.nn.top_k(x,k)
    my_range = tf.expand_dims(tf.range(0, indices.get_shape()[0]), 1)
    my_range_repeated = tf.tile(my_range, [1, k])
    full_indices = tf.concat(2,[tf.expand_dims(my_range_repeated,2), tf.expand_dims(indices, 2)])
    full_indices = tf.reshape(full_indices, [-1, 2])
    to_substract = tf.sparse_to_dense(full_indices, x.get_shape(), tf.reshape(values, [-1]), default_value=0., validate_indices=False)
    res = (x - to_substract)
    return tf.sign(res)



def convertToOneHot(vector, num_classes=5):
    b = np.reshape(vector,[batch_size*(seq_size+1)])
    result = np.zeros(shape=(len(b), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return np.reshape(result.astype(int),[batch_size,seq_size+1,num_classes])


# Input image + Previous Correct Label
x = tf.placeholder(tf.float32,shape=[None, seq_size, input_size + num_classes])
modified_x = tf.transpose(x,[1,0,2])

# Current Correct Label for computing error
y_ = tf.placeholder(tf.float32,shape=[None, seq_size,num_classes]);


read_weights = non_trainable_variable([batch_size, num_rows, num_read_heads],\
										 name = "W_r", init = ['normal', 0.0, 0.1])
usage_weights = non_trainable_variable([batch_size,num_rows],\
										name = "W_u", init = ["constant", 1.0])
memory = non_trainable_variable([batch_size, num_rows, key_size], \
										name = "memory", init = ['normal', 0.0, 1e-4])
error = non_trainable_variable(1, name = "err", init = ["constant", 0.0])
accuracy = non_trainable_variable(1, name = "acc", init = ["constant", 0.0])
accuracy_next_100 = non_trainable_variable(1, name = "acc100", init = ["constant", 0.0])

def counter(y_seq, n = num_classes):
    u,v,w = y_seq.shape
    class_label = np.argmax(y_seq, axis = 2)
    g = np.cumsum(y_seq, axis=1)
    return np.array([g[I][class_label[I]] for I in ndi.ndindex(class_label.shape)]).reshape(u,v)

#################################
#Controller
num_hidden = 200
initializer = tf.random_uniform_initializer(-1,1)
cell = rnn_cell.LSTMCell(num_hidden, initializer=initializer, state_is_tuple=True)

initial_state = cell.zero_state(batch_size, tf.float32)

state = initial_state
num_steps = seq_size
row_index = np.arange(batch_size)

num_instances = tf.placeholder(tf.float32, shape = (batch_size, seq_size))

with tf.variable_scope("RNN"):
    for time_step in range(num_steps):
        if time_step == 0:
    		correct_prediction = []

        if time_step > 0:
            tf.get_variable_scope().reuse_variables()

        # Update LSTM state
        (output, state) = cell(tf.reshape(modified_x[time_step,:,:],shape=[batch_size,input_size+num_classes]), state)

        # Generate write key to store in memory
        # number of write keys = number of read heads
        write_key_temp = linear_layer(output,num_hidden,num_read_heads*key_size, l_name = "write")
        write_key = tf.reshape(write_key_temp,shape=[-1,num_read_heads,key_size])

        # Gate parameter used for computing weights for writing
        alpha = linear_layer(output,num_hidden,num_read_heads, l_name = "interpolator")
        beta = tf.reshape(tf.sigmoid(alpha),shape=[-1,1,num_read_heads])

        # Find which rows are used the least
        least_used_weights = set_k_top_to_zero(usage_weights,num_rows-num_read_heads)

        write_weights = read_weights*beta + tf.reshape(least_used_weights,shape=[-1,num_rows,1])*(1-beta)

        # Erase least used memory row to prevent interference, create space
        memory1 = tf.sub(memory, tf.mul(memory,tf.reshape(least_used_weights,shape=[-1,num_rows,1])))

        # Write the keys generated into memory
        memory = tf.add(memory1, tf.batch_matmul(write_weights,write_key))

        # Generate keys for reading
        read_keys = linear_layer(output,num_hidden,num_read_heads*key_size, l_name = "read")
        keys = tf.reshape(read_keys,shape=[-1,num_read_heads,key_size])

        # Compute Cosine similarity between read key and rows in memory
        key_norm = tf.nn.l2_normalize(keys,2)
        mat_norm = tf.nn.l2_normalize(memory,2)

        similarity = tf.batch_matmul(mat_norm, tf.transpose(key_norm,[0,2,1]))
        temp_similarity = tf.nn.softmax(tf.reshape(tf.transpose(similarity,[0,2,1]),shape=[-1,num_rows]))

        # Read weights are calculated based on the similarity computed previously
        read_weights = tf.transpose(tf.reshape(temp_similarity,shape=[-1,num_read_heads,num_rows]),[0,2,1])

        # Access data from memory
        data_read = tf.batch_matmul(tf.transpose(read_weights,[0,2,1]),memory)
        flattened_data_read = tf.reshape(data_read,shape=[-1,num_read_heads*key_size])

        # Update the usage statistics for the rows of memory
        usage_weights = (weight_decay*usage_weights) + num_read_heads*(tf.reshape(tf.reduce_mean(read_weights+write_weights,2),shape=[-1,num_rows]))

        state_h = state.h #linear_layer(outputs[time_step],num_hidden,num_hidden)

        # Predict class label using data read from memory + LSTM state
        y = (only_linear_layer(flattened_data_read,num_read_heads*key_size,num_classes, l_name = "read2out")
            + only_linear_layer(state_h,num_hidden,num_classes, l_name = "hidden2out"))

        reshaped_y_ = tf.reshape(y_[:,time_step,:],shape=[-1,num_classes])
        column_index = tf.argmax(reshaped_y_,  1)
        t = tf.square(tf.reshape(num_instances[:,time_step], shape=[batch_size]))

        scale_factor = t/tf.to_float(tf.reduce_sum(t)) * seq_size/num_classes
        error += tf.reduce_mean(scale_factor*tf.nn.softmax_cross_entropy_with_logits(y, reshaped_y_))

        correct_prediction.append(tf.equal(tf.argmax(y, 1),column_index))#reshaped_y_,1))#


out_prediction = tf.pack(correct_prediction)
train_step = tf.train.AdamOptimizer(1e-3).minimize(error)

saver = tf.train.Saver(max_to_keep=200)

##############################################################################
# Training
sess = tf.Session()
sess.run(tf.initialize_all_variables())
avg_err = 0
shuffle_vector=range(num_classes)

batch_list = os.listdir("/u/training/tra263/scratch/ntm-one-shot/OmniglotBatches")
num_episodes = 100000
for episode in range(num_episodes):
     if episode%500 == 0:
         save_path=saver.save(sess,"./newaccuracy_ckpts/model_"+str(episode)+".ckpt")
     filename = np.random.choice(batch_list)
     with np.load("/u/training/tra263/scratch/ntm-one-shot/OmniglotBatches/"+filename) as data:
         batch_xs = data['ip']
         example_output = data['op']
     batch_ys = convertToOneHot(np.reshape(example_output,[batch_size*(seq_size+1)]))
     in_x = np.concatenate((batch_xs[:,1:seq_size+1],batch_ys[:,:seq_size]),axis=2)
     in_y = batch_ys[:,1:]
     acc_list, err, step = sess.run([out_prediction, error, train_step], feed_dict={x:in_x,
								     y_:in_y,
								     num_instances:counter(in_y)})
     acc_matrix = acc_list.mean(axis = 0)
     avg_accuracy_eos = np.mean(acc_matrix[-num_classes:])
     avg_err += err
     if episode%100 == 0:
         print "Episode: " + str(episode) + " EOS accuracy " + str(avg_accuracy_eos) + " error " + str(err) + " avg error " + str(avg_err/(episode+1))


#################################################################################
# Testing
#saver.restore(sess,'/u/training/tra263/scratch/ntm-one-shot/model_.ckpt')
A= np.zeros(30)
count = np.zeros(30)
num_episodes = 50
for episode in range(num_episodes):
    filename = np.random.choice(batch_list)
    with np.load("/u/training/tra263/scratch/ntm-one-shot/OmniglotBatches/"+filename) as data:
        batch_xs = data['ip']
        example_output = data['op']
#    i, (batch_xs,example_output) = generator.next()
    batch_ys = convertToOneHot(np.reshape(example_output,[batch_size*(seq_size+1)]))
    in_x = np.concatenate((batch_xs[:,1:seq_size+1],batch_ys[:,:seq_size]),axis=2)
    in_y = batch_ys[:,1:]
    vec, acc, err = sess.run([correct_prediction, accuracy_vector, error], feed_dict={x:in_x,
								     y_:n_y,
								     num_instances:counter(in_y)})
    temp = np.asarray(acc)
    classes = np.argmax(batch_ys[0,1:],1)
    for class_num in range(num_classes):
        ind = np.argwhere(classes==class_num)
        r = temp[ind]
        for j in range(min(r.size,30)):
            A[j] += r[j]
            count[j] += 1

plt.plot(np.divide(A,count)*100)
plt.xlabel("Class instance")
plt.ylabel("accuracy")
plt.savefig("Accuracy.png", format = 'png')
plt.show()

w = np.divide(A,count)*100

fig, ax1 = plt.subplots()
ax1.plot(w, 'b-')
ax1.set_xlabel('Class instance')
# Make the y-axis label and tick labels match the line color.
ax1.set_ylabel('Accuracy', color='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b')

ax2 = ax1.twinx()
ax2.plot(count, 'r.')
ax2.set_ylabel('count', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
plt.savefig("Accuracy_and_count.png", format = 'png')
plt.show()

np.savetxt('class_accuracy.txt', w, delimiter = ',')

sess.close()
