import deepxde as dde
import numpy as np
import os
from prm import n_dt, n_test_cub, niter, n_test, dt, stepoch, opt1, lossw1_a, lossw1_b
from prm import opt2, lossw2_a, lossw2_b, n_train1, n_train2
from prm import lrate1_a, lrate1_b, nepochs1_a, nepochs1_b, nepochs2_a, nepochs2_b, nepochs1
import prm
import torch
from torch import nn
from numpy_to_torch import pytorch_output
import time

def run_iterations():

    import model_nn
    import plots1
    import results
    import store_data

    # Create testing points in a structured mesh
    xyz = results.create_mesh()

    for prm.it in range(niter):

        print()
        print("ITERATION", prm.it)
        print()

        # Train for t=0
        (data_a, model_a) = model_nn.create_model_a(prm.it)
        prm.time = 0
        prm.time_step = 0
        print("\nTIME = ", prm.time, "\n")

        # 1st stage training
        k = 0.1**(1/10000)
        model_a.compile(opt1, lr=lrate1_a, loss_weights=lossw1_a, decay=("exponential", k))
        model_a.train(iterations=nepochs1_a, display_every=stepoch)

        previous_model = 0	# No previous model at t=0

        prm.ib = 0

        # Train for t>0. This loop runs once per time step, producing one model per time step
        n_dt1 = n_dt - 1
        for prm.ib in range(n_dt1):
            print("\nITERATIONSSS_IB = ", prm.ib, "\n")
            if prm.ib == 0:
                previous_model = model_a
            else:
                previous_model = model_b	# the model_b at the past time step

            # A new model is created, with new parameters. To use the ones from the previous model, see lines below.
            (data_b, model_b) = model_nn.create_model_b(previous_model, prm.it)

            # Copy the weights and biases from the previous net to the new net
            # This ensures model_b starts exactly where the previous model finished (transfer learning)
            model_b.net.load_state_dict(previous_model.net.state_dict())

            prm.time = (prm.ib+1)*dt
            prm.time_step = prm.ib + 1
            print("\nTIME = ", prm.time, " | ", "TIME STEP: ", prm.time_step, "\n")

            # 1st stage training
            k_b = 0.1 ** (1/300)
            model_b.compile(opt1, lr=lrate1_b, loss_weights=lossw1_b, decay=("exponential", k_b))
            model_b.train(iterations=nepochs1_b, display_every=stepoch)

    return niter
