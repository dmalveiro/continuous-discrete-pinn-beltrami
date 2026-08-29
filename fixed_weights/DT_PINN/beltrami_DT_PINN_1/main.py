import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"	# only run on GPU n (comment if using Horovod or system with single GPU)
os.environ['DDE_BACKEND'] = 'pytorch'
import torch
import deepxde as dde

# Set default float to float64 (double precision)
dde.config.set_default_float("float32")

import iterationsss
nrun = iterationsss.run_iterations()
