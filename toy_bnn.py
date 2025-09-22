import tensorflow as tf
import os
print(os.getcwd())

from src.dicl.rl.tf_models import *
from src.dicl.rl.tf_models.constructor import *

import numpy as np

import tensorflow.compat.v1 as tf

import time

tf.disable_v2_behavior()

bnn = construct_shallow_model(obs_dim = 3, act_dim = 1, hidden_dim = 100, num_networks = 1, num_elites =  1)

s =  np.ones((1,4))
s1 = np.ones((1,3))
time1 = time.time()
bnn.train(s, s1, epochs = 1000)
time2 = time.time()


print("PRANK ", s)
print("TIME ", time2 - time1)
