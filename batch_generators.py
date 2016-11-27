# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:27:24 2016

@author: konik
"""

import os
import numpy as np
from mann.utils.generators import OmniglotGenerator
import time
import multiprocessing as mp


def f(batch_num):
   g = OmniglotGenerator(data_folder='/media/konik/Misc/OneShot/data/omniglot/images_background/', \
   batch_size=16, nb_samples=5, nb_samples_per_class=10, max_rotation=-np.pi/6.0, \
   max_shift=0.01, max_iter=None)
   _, (eg_input, eg_output) = g.next()
   np.savez("newbatch"+str(batch_num), ip = eg_input, op = eg_output)



if __name__ == '__main__':
    t1 = time.time()
    try:
        nprocesses = 100
        pool = mp.Pool(nprocesses)

        result = pool.map_async(f, np.arange(6000))
        pool.close()
        pool.join()
        result.get()
    except KeyboardInterrupt:
        print " Time elapsed %0.5f" %(time.time() - t1)
    finally:
        print " Time elapsed %0.5f" %(time.time() - t1)

#filename = "batch"
#
#generator = OmniglotGenerator(data_folder='/media/konik/Misc/OneShot/data/omniglot/images_background/', batch_size=16, \
#nb_samples=5, nb_samples_per_class=10, max_rotation=-np.pi/6.0, max_shift=0.01, max_iter=None)
#
#c = 2
#
#for i in xrange(c):
#    _, (eg_input, eg_output) = generator.next()
#    np.savez(filename+str(i), ip = eg_input, op = eg_output)
#
#
#t1 = time.time()
#
#generator.next()
#
#print("Generator took %0.5fs" % (time.time() - t1))
#
#t1 = time.time()
#with np.load(filename+"0.npz") as data:
#    ip = data['ip']
#    op = data['op']
#
#print("Batch loading took %0.5fs" % (time.time() - t1))


