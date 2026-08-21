import random
import torch
import numpy as np

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

# Auxiliary variables
count = 0
gg = 0

# Space/time domain limits
x_min = [-1, -1, -1]
x_max = [1, 1, 1]
t_min = 0
t_max = 1

# Current time instant, time step and number of time steps
time = 0
ib = 0
time_step = 0
dt = 0.01
n_dt = int(1 + 1 / dt)
it = 0		# iteration (run)

# Training points
n_train1 = 800          		   # Number of training points at t>0
n_train2 = 200	                   # Number of training points at the boundaries

# Testing points (model testing to assess L2 errors between exact and predictions)
n_test = 32
n_test_cub = n_test**3

# Validation points
n_val = 1000

# Neural network parameters
inl = 3				# number of elements at the input layer
n_hidl = 10			# number of hidden layers
n_elem = 200    		# number of elements per hidden layer
outl = 4			# number of elements at the output layer
act_func = "silu"		# activation function
w_init = "Glorot normal"	# weight initializer method

# Number of iterations to run the model
niter = 10

# Model compilation and training parameters
# 1st stage
opt1 = "adam"						# optimizer
lrate1_a = 1e-3						# learning rate
lrate1_b = 1e-5
a1 = 1							# weight attributed to the residuals of the NS equations
a2 = 1                                            	# weight attributed to the boundary conditions
a12 = 1
lossw1_a = [1, 1, 1, 1]
lossw1_b = [a1, a1, a1, a12, a2, a2, a2, a2]
nepochs1_a = 60000					# number of epochs per iteration
nepochs1_b = 300
# 2nd stage (optional)
opt2 = "L-BFGS"
lossw2_a = [1, 1, 1, 1]
lossw2_b = [a1, a1, a1, a12, a2, a2, a2, a2]
nepochs2_a = 200
nepochs2_b = 200

# Total number of epochs (only 1st stage)
nepochs1 = nepochs1_a + (n_dt - 1) * nepochs1_b

# Print losses in intervals of 'stepoch' epochs
stepoch = 10

# Save additional files (0: no; 1: yes)
save_additional = 0
