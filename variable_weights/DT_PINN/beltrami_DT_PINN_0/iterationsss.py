import deepxde as dde
import numpy as np
import os
from prm import n_dt, n_test_cub, nrun, n_test, dt, step_iter, opt1, lossw1_a, lossw1_b
from prm import opt2, lossw2_a, lossw2_b, n_train1, n_train2
from prm import lrate1_a, lrate1_b, niters1_a, niters1_b, niters2_a, niters2_b, niters1
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

    for prm.r in range(nrun):

        # Train for t=0
        (data_a, model_a) = model_nn.create_model_a(prm.r)
        prm.time = 0
        prm.time_step = 0
        print("\nRUN: ", prm.r, " | ", "TIME = ", prm.time, " | ", "TIME STEP: ", prm.time_step, " -------------------------------\n")
#        plots1.printall(data_a, prm.time_step, prm.r)

        # 1st stage training
        k = 0.1**(1/10000)
        model_a.compile(opt1, lr=lrate1_a, loss_weights=lossw1_a, decay=("exponential", k))
        start = time.perf_counter()
        losshistory, train_state = model_a.train(iterations=niters1_a, display_every=step_iter)
        elapsed = time.perf_counter() - start

        # 2nd stage training (L-BFGS) - optional
#        dde.optimizers.config.LBFGS_options["iter_per_step"] = step_iter
#        dde.optimizers.config.set_LBFGS_options(maxcor=100, ftol=0, gtol=1e-08, maxiter=niters2_a, maxfun=None, maxls=50)
#        model_a.compile(opt2, loss_weights=lossw2_a)
#        losshistory, train_state = model_a.train()

        # Save loss history
        dde.saveplot(losshistory, train_state, issave=True, isplot=True)
        os.rename("loss.dat", f"loss_run{prm.r}_ts{prm.time_step:04d}.dat")

        previous_model = 0	# No previous model at t=0

        # Predict results for t=0 at the test set
        (l2_difference_u, l2_difference_v, l2_difference_w, l2_difference_p, residual,
        u_pred, v_pred, w_pred, p_pred_corrected, u_exact, v_exact, w_exact, p_exact, f
        ) = results.pred_exact(xyz, previous_model, model_a, prm.time_step)

        print()
        print("RUN", prm.r)
        print("Accuracy at t = ", prm.time)
        print("Mean residual:", residual)
        print("L2 relative error in u:", l2_difference_u)
        print("L2 relative error in v:", l2_difference_v)
        print("L2 relative error in w:", l2_difference_w)
        print("L2 relative error in p:", l2_difference_p)
        print()

        prm.ib = 0

        # Store training time
        store_data.save_training_times(prm.ib, prm.r, elapsed, nrun, prm.time_step)

        # Store the data produced at 'results.pred_exact' in csv files
        store_data.in_variables(prm.ib, prm.r, xyz,
                                l2_difference_u, l2_difference_v, l2_difference_w, l2_difference_p, residual,
                                u_pred, v_pred, w_pred, p_pred_corrected, u_exact, v_exact, w_exact, p_exact, f, prm.time_step)

        # Train for t>0. This loop runs once per time step, producing one model per time step
        n_dt1 = n_dt - 1
        for prm.ib in range(n_dt1):
            if prm.ib == 0:
                previous_model = model_a
            else:
                previous_model = model_b	# the model_b at the past time step

            # A new model is created, with new parameters. To use the ones from the previous model, see lines below.
            (data_b, model_b) = model_nn.create_model_b(previous_model, prm.r)

            # Copy the weights and biases from the previous net to the new net
            # This ensures model_b starts exactly where the previous model finished (transfer learning)
            model_b.net.load_state_dict(previous_model.net.state_dict())

            prm.time = (prm.ib+1)*dt
            prm.time_step = prm.ib + 1
            print("\nRUN: ", prm.r, " | ", "TIME = ", prm.time, " | ", "TIME STEP: ", prm.time_step, " -------------------------------\n")
#            plots1.printall(data_b, prm.time_step, prm.r)

            # 1st stage training
            k_b = 0.1 ** (1/300)
            model_b.compile(opt1, lr=lrate1_b, loss_weights=lossw1_b, decay=("exponential", k_b))
            start = time.perf_counter()
            losshistory, train_state = model_b.train(iterations=niters1_b, display_every=step_iter)
            elapsed = time.perf_counter() - start

            # 2nd stage training (L-BFGS) - optional
#            dde.optimizers.config.LBFGS_options["iter_per_step"] = step_iter
#            dde.optimizers.config.set_LBFGS_options(maxcor=100, ftol=0, gtol=1e-08, maxiter=niters2_b, maxfun=None, maxls=50)
#            model_b.compile(opt2, loss_weights=lossw1_b)
#            losshistory, train_state = model_b.train()

            # Save loss history
            dde.saveplot(losshistory, train_state, issave=True, isplot=True)
            os.rename("loss.dat", f"loss_run{prm.r}_ts{prm.time_step:04d}.dat")

            # Predict results for t>0, for each t, at the test set
            (l2_difference_u, l2_difference_v, l2_difference_w, l2_difference_p, residual,
            u_pred, v_pred, w_pred, p_pred_corrected, u_exact, v_exact, w_exact, p_exact, f
            ) = results.pred_exact(xyz, previous_model, model_b, prm.time_step)

            print()
            print("RUN", prm.r)
            print("Accuracy at t = ", prm.time)
            print("Mean residual:", residual)
            print("L2 relative error in u:", l2_difference_u)
            print("L2 relative error in v:", l2_difference_v)
            print("L2 relative error in w:", l2_difference_w)
            print("L2 relative error in p:", l2_difference_p)
            print()

            # Store training time
            store_data.save_training_times(prm.ib, prm.r, elapsed, nrun, prm.time_step)

            # Store the data produced at 'results.pred_exact' in csv files
            store_data.in_variables(prm.ib, prm.r, xyz,
                                    l2_difference_u, l2_difference_v, l2_difference_w, l2_difference_p, residual,
                                    u_pred, v_pred, w_pred, p_pred_corrected, u_exact, v_exact, w_exact, p_exact, f, prm.time_step)

    return nrun
