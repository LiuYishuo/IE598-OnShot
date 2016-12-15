# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:27:24 2016

@author: konik
"""

import numpy as np
import time
import multiprocessing as mp
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)

def f(batch_num):
    count = 0
    eg_input = []
    eg_output = []
    while (count < 50):
        x, y = mnist.train.next_batch(1)
        if y < 5:
            count += 1
            eg_input.append(x)
            eg_output.append(y)
    print eg_input.shape
    np.savez("newbatch"+str(batch_num), ip = eg_input, op = eg_output)



if __name__ == '__main__':
    t1 = time.time()
    try:
        # spawn processes on OS
        nprocesses = 50
        pool = mp.Pool(nprocesses)
        # allow asynchronous batch generation, enter the number of batches
        # you want as argument
        result = pool.map_async(f, np.arange(20000))
        pool.close()
        pool.join()
        result.get()
    except KeyboardInterrupt:
        print " Time elapsed %0.5f" %(time.time() - t1)
    finally:
        print " Time elapsed %0.5f" %(time.time() - t1)
