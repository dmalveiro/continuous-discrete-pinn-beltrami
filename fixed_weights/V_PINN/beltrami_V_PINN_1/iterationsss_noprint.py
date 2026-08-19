import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os
from param import n_dt, n_test_cub, niter, n_test, dt, nepochs1, stepoch, opt1, lrate1, lossw1, opt2, lossw2, nepochs2
import torch
from torch import nn
import time

def run_iterations():

    import model_nn

    for it in range(niter):

        print()
        print("ITERATION", it)
        print()

        # Create and compile the NN model
        (data, model) = model_nn.create_model(it)

        # 1st stage compiling and training
        k = 0.1**(1/10000)
        model.compile(opt1, lr=lrate1, loss_weights=lossw1, decay=("exponential", k))

        # Set the maximum training time (in minutes)
        timer_callback = dde.callbacks.Timer(available_time=85.81)

        model.train(iterations=nepochs1, display_every=stepoch, callbacks=[timer_callback])
#        losshistory, train_state = model.train(iterations=nepochs1, display_every=stepoch)

        # 2nd stage training (L-BFGS) - optional
#        dde.optimizers.config.LBFGS_options["iter_per_step"] = stepoch
#        dde.optimizers.config.set_LBFGS_options(maxcor=100, ftol=0, gtol=1e-08, maxiter=nepochs2, maxfun=None, maxls=50)
#        model.compile(opt2, loss_weights=lossw2)
#        losshistory, train_state = model.train()

    return niter
