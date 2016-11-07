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
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)


image_size = 28
input_size = image_size*image_size
key_size = 50
num_rows = 50
weight_decay = 0.8
num_classes = 10

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


x = tf.placeholder(tf.float32,shape=[1,input_size])
prev_output = tf.placeholder(tf.float32,shape=[1,num_classes])
y_ = tf.placeholder(tf.float32,shape=[1,num_classes]);

memory_module = tf.placeholder(tf.float32,[num_rows, key_size])

#################################
output_size = 2*key_size + 1 # one output for gate parameter

combined_input = tf.concat(1,[x,prev_output])
# Controller - 2 layer feed forward network
x1 = linear_layer(combined_input,input_size+num_classes,500)
x2 = linear_layer(x1,500,200)
x3 = linear_layer(x2,200,output_size)

read_key = tf.slice(x3,[0,0],[1,key_size])
write_key = tf.slice(x3,[0,key_size],[1,key_size])
alpha = tf.slice(x3,[0,2*key_size],[1,1]) # gate parameter

key_norm = tf.nn.l2_normalize(read_key,1)


######################################################################
# least used policy to be implemented
previous_read_weights = tf.placeholder(tf.float32,shape=[num_rows,1])
least_used_weights = tf.placeholder(tf.float32,shape=[num_rows,1])


beta = tf.sigmoid(alpha)
write_weights = previous_read_weights*beta + least_used_weights*(1-beta)

previous_usage_weights = tf.placeholder(tf.float32,shape=[num_rows,1])

memory_erase = tf.placeholder(tf.float32,shape=[num_rows,1])
memory = tf.add(tf.mul(memory_erase,memory_module),
                                        tf.matmul(write_weights,write_key))


mat_norm = tf.nn.l2_normalize(memory,1)
similarity = tf.matmul(mat_norm, tf.transpose(key_norm))
read_weights = tf.transpose(tf.nn.softmax(tf.transpose(similarity)))
data_read = tf.matmul(tf.transpose(read_weights),memory)

input_to_softmax = linear_layer(tf.concat(1,[data_read, x2]),key_size+200,100)
y = only_linear_layer(input_to_softmax,100,num_classes)

#input_to_softmax = (tf.concat(1,[data_read, x2]))
#y = only_linear_layer(input_to_softmax,key_size+200,num_classes)

usage_weights = (weight_decay*previous_usage_weights) + (
                                                read_weights + write_weights)


error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
train_step = tf.train.AdamOptimizer(1e-4).minimize(error)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())

initial_usage_weights = tf.truncated_normal([num_rows,1],stddev=0.1)
stored_usage_weights = sess.run(initial_usage_weights)

initial_read_weights = tf.truncated_normal([num_rows,1],stddev=0.1)
stored_read_weights = sess.run(initial_read_weights)

initial_memory = tf.truncated_normal([num_rows, key_size],stddev=0.1)
stored_memory = sess.run(initial_memory)

stored_least_used_weights = sess.run(initial_usage_weights)
memory_erase_vector = np.ones([num_rows,1])

initial_error = tf.constant(1.0,shape=[1])
current_error = sess.run(initial_error)

batch_xs1, batch_ys1 = mnist.train.next_batch(1)
output = np.zeros([1,num_classes])
avg_acc = 0


num_episodes = 50000
for episode in range(num_episodes):
    random.shuffle(shuffle_vector)
    print episode
    if episode%500 == 0:
        save_path=saver.save(sess,"model-"+str(episode)+".ckpt")

    stored_usage_weights = sess.run(initial_usage_weights)
    stored_read_weights = sess.run(initial_read_weights)
    stored_memory = sess.run(initial_memory)
    stored_least_used_weights = sess.run(initial_usage_weights)
    memory_erase_vector = np.ones([num_rows,1])



    accuracy_vector = np.zeros((num_classes,1))
    count_vector = np.zeros((num_classes,1))

    rotation = np.random.randint(-10,10,100)
    for i in range(100):
        in_x, original_class = mnist.train.next_batch(1)

        batch_xs = np.reshape(ndimage.rotate(np.reshape(in_x,[image_size,image_size]), rotation[i], reshape=False),[1,input_size])

        count_vector[original_class[0]] = count_vector[original_class[0]]+1
        in_y = shuffle_vector[original_class[0]]
        batch_ys = np.zeros([1,num_classes])
        batch_ys[0,in_y] = 1

        output, stored_read_weights,stored_usage_weights,stored_memory, step, acc = sess.run(
                                        [y_, read_weights, usage_weights, memory, train_step, accuracy],
                                        feed_dict=
                                        {x: batch_xs, y_: batch_ys,
                                        prev_output: output,
                                        memory_module: stored_memory,
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

        accuracy_vector[original_class[0]] = accuracy_vector[original_class[0]] + acc

        avg_acc = avg_acc + acc
#    if episode%10 == 0:
#        print (accuracy_vector/count_vector)
    print avg_acc/(episode+1)
print avg_acc/num_episodes

#saver.restore(sess,'model-1000.ckpt')
stored_usage_weights = sess.run(initial_usage_weights)
stored_read_weights = sess.run(initial_read_weights)
stored_memory = sess.run(initial_memory)
stored_least_used_weights = sess.run(initial_usage_weights)
memory_erase_vector = np.ones([num_rows,1])
total_acc = np.zeros(500)

output = np.zeros([1,num_classes])
for i in range(500):

    batch_xs, in_y = mnist.test.next_batch(1)
    in_y = shuffle_vector[in_y[0]]
    batch_ys = np.zeros([1,num_classes])
    batch_ys[0,in_y] = 1

    output, stored_read_weights,stored_usage_weights,stored_memory, acc = sess.run(
                                    [y, read_weights, usage_weights, memory, accuracy],
                                    feed_dict=
                                    {x: batch_xs, y_: batch_ys,
                                    prev_output: output,
                                    memory_module: stored_memory,
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
    total_acc[i] = acc

sess.close()
print np.sum(total_acc)
