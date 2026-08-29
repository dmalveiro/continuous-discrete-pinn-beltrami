import numpy as np
import prm
from prm import n_test_cub, nrun, n_dt, save_additional
import os
import pandas as pd

# Matrix to save the L2 relative errors for 5 variables (u,v,w,p,res) per time step and per run r
final_data = np.zeros((n_dt*5, nrun))

uvwp_pred = np.zeros((n_test_cub*n_dt, 4))      # 4 columns: (u,v,w,p) predicted
uvwp_exact = np.zeros((n_test_cub*n_dt, 4))     # 4 columns: (u,v,w,p) exact
rel_error = np.zeros((n_test_cub*n_dt, 4))      # rel_error(i,r) = |uvwp_pred(i,r)-uvwp_exact(i,r)| / |uvwp_exact(i,r)|
residualss = np.zeros((n_test_cub*n_dt, 4))     # residuals of the NS equations
xyz_all = np.zeros((n_test_cub*n_dt, 3))        # 3 columns: test set: all these at the coordinates xyz_all=(x,y,z)

# Matrix to save the training times per time step and per run r
training_times = np.zeros((n_dt, nrun))

def in_variables(i, r, xyz,
                 l2_difference_u, l2_difference_v, l2_difference_w, l2_difference_p, residual,
                 u_pred, v_pred, w_pred, p_pred_corrected, u_exact, v_exact, w_exact, p_exact, f, time_step):

    # At each time step, store the values

    if time_step > 0: i=i+1

    c1 = 5*(i+1)
    c2b = i*n_test_cub
    c2e = (i+1)*n_test_cub
    c3 = 5
    res_l2 = [residual, l2_difference_u, l2_difference_v, l2_difference_w, l2_difference_p]
    pred = [u_pred, v_pred, w_pred, p_pred_corrected]
    exact = [u_exact, v_exact, w_exact, p_exact]

    k = 0
    c4 = 0
    for k in range(c3):
        c4 = c3-k
        final_data[c1-c4, r] = res_l2[k]
        uvwp_pred[c2b:c2e, k-1] = pred[k-1]
        uvwp_exact[c2b:c2e, k-1] = exact[k-1]
        rel_error[c2b:c2e, k-1] = abs(pred[k-1] - exact[k-1]) / abs(exact[k-1])

    residualss[c2b:c2e] = f
    xyz_all[c2b:c2e, :] = xyz

    # Save residuals and L2 error averages in a .csv file
    np.savetxt('final_data.csv', final_data, delimiter=',')

    # Save additional data in .csv files
    if save_additional == 1:
        np.savetxt(f'uvwp_pred_run{r:04d}.csv', uvwp_pred, delimiter=',')
        np.savetxt(f'uvwp_exact_run{r:04d}.csv', uvwp_exact, delimiter=',')
        np.savetxt(f'rel_error_run{r:04d}.csv', rel_error, delimiter=',')
        np.savetxt(f'residualss_run{r:04d}.csv', residualss, delimiter=',')
        np.savetxt(f'xyz_all_run{r:04d}.csv', xyz_all, delimiter=',')

# Write and save training times
def save_training_times(i, r, elapsed, nrun, time_step):

    if time_step > 0: i=i+1

    # Row i: time step; column r: run
    training_times[i, r] = elapsed

    # Save training times
    print("\nSaving training times...\n")
    np.savetxt('training_times.csv', training_times, delimiter=',')



