#import horovod.torch as hvd
import numpy as np
import torch
import random

# Defining GPU device as "device"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Precision
precision = torch.float32

# Deterministic code
deterministic = 1
if deterministic == 1:
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Parameters for analytical solutions
a = 1
d = 1
a = torch.tensor(a, dtype=torch.int32)
d = torch.tensor(d, dtype=torch.int32)

# Reynolds number
Re = 1

# Space/time domain limits
x_min = [-1, -1, -1]
x_max = [1, 1, 1]
t_min = 0
t_max = 1

# Time step and number of time steps
dt = 0.01
n_dt = int(1 + 1 / dt)
count = 0

# Training points
n_train1 = 700                      # Number of training points at t>0
n_train2 = 150                      # Number of training points at the boundaries
n_train3 = 150 		            # Number of training points at t=0

# Testing points (model testing to assess L2 errors between exact and predictions)
n_test = 32
n_test_cub = n_test**3

# Validation points
n_val = 700

# Neural network parameters
inl = 4				# number of elements at the input layer
n_hidl = 10			# number of hidden layers
n_elem = 200    		# number of elements per hidden layer
outl = 4			# number of elements at the output layer
act_func = "tanh"		# activation function
w_init = "Glorot normal"	# weight initializer method

# Number of runs
nrun = 10

# Model compilation and training parameters
# 1st stage
opt1 = "adam"						# optimizer
lrate1 = 1e-5						# learning rate
a1 = 1							# weight attributed to the residuals of the NS equations
a2 = 1                                            	# weight attributed to the boundary conditions
a3 = 1							# weight attributed to the initial conditions
lossw1 = [a1, a1, a1, a1, a2, a2, a2, a3, a3, a3]	# loss weights
niters1 = 10000000					# number of iterations per run
# 2nd stage
opt2 = "L-BFGS"
lossw2 = [a1, a1, a1, a1, a2, a2, a2, a3, a3, a3]
niters2 = 1000

# Print losses in intervals of 'step_iter' iterations
step_iter = 10

# Save additional files (0: no; 1: yes)
save_additional = 0
