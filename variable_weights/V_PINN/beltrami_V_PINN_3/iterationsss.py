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
    import plots1
    import results
    import store_data

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
        timer_callback = dde.callbacks.Timer(available_time=91.89)

        start = time.perf_counter()
        losshistory, train_state = model.train(iterations=nepochs1, display_every=stepoch, callbacks=[timer_callback])
#        losshistory, train_state = model.train(iterations=nepochs1, display_every=stepoch)
        elapsed = time.perf_counter() - start

        # 2nd stage training (L-BFGS) - optional
#        dde.optimizers.config.LBFGS_options["iter_per_step"] = stepoch
#        dde.optimizers.config.set_LBFGS_options(maxcor=100, ftol=0, gtol=1e-08, maxiter=nepochs2, maxfun=None, maxls=50)
#        model.compile(opt2, loss_weights=lossw2)
#        losshistory, train_state = model.train()

        # Save loss history
        dde.saveplot(losshistory, train_state, issave=True, isplot=True)
        os.rename("loss.dat", f"loss_{it}.dat")

        # Store training time
        store_data.save_training_times(it, elapsed, niter)

        # Create a n_test^3 mesh, with uniformly spaced mesh points
        # x=y=z=(n_test,n_test,n_test)
        x, y, z = np.meshgrid(
            np.linspace(-1, 1, n_test), np.linspace(-1, 1, n_test), np.linspace(-1, 1, n_test)
        )

        # Matrix (n_test^3,3): each line is the (x,y,z) coordinates of a mesh point
        X = np.vstack((np.ravel(x), np.ravel(y), np.ravel(z))).T

        for i in range(n_dt):

            # Vectors with time instants t
            t = np.ones(n_test_cub).reshape(n_test_cub, 1)*i*dt

            # Merge spacial and temporal coordinates
            # These are the coordinates for the test set
            xt = np.hstack((X, t))       # (x,y,z,t)

            # Optional: save test set in a csv file
            #np.savetxt('xt_%d.csv' % i, xt, delimiter=',')

            # from results import residual, l2_difference_u, l2_difference_v, l2_difference_w, l2_difference_p
            (l2_difference_u, l2_difference_v, l2_difference_w, l2_difference_p, residual,
            u_pred, v_pred, w_pred, p_pred_corrected, u_exact, v_exact, w_exact, p_exact, f
            ) = results.pred_exact(xt, model)

            print()
            print("Accuracy at t = ", i*dt)
            print("Mean residual:", residual)
            print("L2 relative error in u:", l2_difference_u)
            print("L2 relative error in v:", l2_difference_v)
            print("L2 relative error in w:", l2_difference_w)
            print("L2 relative error in p:", l2_difference_p)
            print()

            store_data.in_variables(i, it, xt,
                                    l2_difference_u, l2_difference_v, l2_difference_w, l2_difference_p, residual,
                                    u_pred, v_pred, w_pred, p_pred_corrected, u_exact, v_exact, w_exact, p_exact, f)

    return niter
