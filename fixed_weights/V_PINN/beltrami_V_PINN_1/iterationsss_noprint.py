import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
from param import n_dt, n_test_cub, nrun, n_test, dt, niters1, step_iter, opt1, lrate1, lossw1, opt2, lossw2, niters2
import torch
from torch import nn
import time

def run_iterations():

    import model_nn

    for r in range(nrun):

        print()
        print("\nRUN: ", r, " -------------------------------\n")
        print()

        # Create and compile the NN model
        (data, model) = model_nn.create_model(r)

        # 1st stage compiling and training
        k = 0.1**(1/10000)
        model.compile(opt1, lr=lrate1, loss_weights=lossw1, decay=("exponential", k))

        # Set the maximum training time (in minutes)
        timer_callback = dde.callbacks.Timer(available_time=85.81)

        model.train(iterations=niters1, display_every=step_iter, callbacks=[timer_callback])
#        losshistory, train_state = model.train(iterations=niters1, display_every=step_iter)

        # 2nd stage training (L-BFGS) - optional
#        dde.optimizers.config.LBFGS_options["iter_per_step"] = step_iter
#        dde.optimizers.config.set_LBFGS_options(maxcor=100, ftol=0, gtol=1e-08, maxiter=niters2, maxfun=None, maxls=50)
#        model.compile(opt2, loss_weights=lossw2)
#        losshistory, train_state = model.train()

    return nrun
