# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:27:24 2016

@author: konik
"""

import os
import numpy as np
import scipy.misc
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

batch_size = 50*16

def f(batch_num):
    """Generates a batch with an appended batch_number"""
    count = 0
    eg_input = []
    eg_output = []
    while (count < batch_size):
        x, y = mnist.train.next_batch(1)
        if y < 1:
            count += 1
            im = np.reshape(x,[28,28])
            cropped_im = scipy.misc.imresize(im,[20,20])
            x = np.reshape(cropped_im,[20*20])
            eg_input.append(x)
            eg_output.append(y)
    eg_input = np.asarray(eg_input).reshape((16, 50, 20*20))
    eg_output = np.asarray(eg_output).reshape((16, 50))
    np.savez("./MNIST_batches/newbatch"+str(batch_num), ip = eg_input, op = eg_output)



if __name__ == '__main__':
    t1 = time.time()
    try:
        # spawn 24 processes on the OS
        nprocesses = 24
        pool = mp.Pool(nprocesses)
        # allow asynchronous batch-processing
        result = pool.map_async(f, np.arange(60))
        pool.close()
        pool.join()
        result.get()
    except KeyboardInterrupt:
        print " Time elapsed %0.5f" %(time.time() - t1)
    finally:
        print " Time elapsed %0.5f" %(time.time() - t1)

