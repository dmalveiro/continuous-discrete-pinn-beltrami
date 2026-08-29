import numpy as np
from param import n_test_cub, nrun, n_dt, save_additional

# Matrix to save the L2 relative errors for 5 variables (u,v,w,p,res) per time step and per run r
final_data = np.zeros((n_dt*5, nrun))

uvwp_pred = np.zeros((n_test_cub*n_dt, 4))      # 4 columns: (u,v,w,p) predicted
uvwp_exact = np.zeros((n_test_cub*n_dt, 4))     # 4 columns: (u,v,w,p) exact
rel_error = np.zeros((n_test_cub*n_dt, 4))      # rel_error(i,r) = |uvwp_pred(i,r)-uvwp_exact(i,r)| / |uvwp_exact(i,r)|
residualss = np.zeros((n_test_cub*n_dt, 4))     # residuals of the NS equations
xt_all = np.zeros((n_test_cub*n_dt, 4))         # 4 columns: test set: all these at the coordinates xt_all=(x,y,z,t)

# Matrix to save the training times per time step and per run r
training_times = np.zeros((1, nrun))

def in_variables(i, r, xt,
                 l2_difference_u, l2_difference_v, l2_difference_w, l2_difference_p, residual,
                 u_pred, v_pred, w_pred, p_pred_corrected, u_exact, v_exact, w_exact, p_exact, f):

    # At each time step, store the values

    c1 = 5*(i+1)
    c2b = i*n_test_cub
    c2e = (i+1)*n_test_cub
    c3 = 5
    res_l2 = [residual, l2_difference_u, l2_difference_v, l2_difference_w, l2_difference_p]
    pred = [u_pred, v_pred, w_pred, p_pred_corrected]
    exact = [u_exact, v_exact, w_exact, p_exact]

    k=0
    for k in range(c3):
        c4 = c3-k
        final_data[c1-c4, r] = res_l2[k]
        uvwp_pred[c2b:c2e, k-1] = pred[k-1]
        uvwp_exact[c2b:c2e, k-1] = exact[k-1]
        rel_error[c2b:c2e, k-1] = abs(pred[k-1] - exact[k-1]) / abs(exact[k-1])

    residualss[c2b:c2e] = f
    xt_all[c2b:c2e, :] = xt

    # Save residuals and L2 error averages in a .csv file
    np.savetxt(f'final_data.csv', final_data, delimiter=',')

    # Save additional data in .csv files
    if save_additional == 1:
        np.savetxt(f'uvwp_pred_run{r}.csv', uvwp_pred, delimiter=',')
        np.savetxt(f'uvwp_exact_run{r}.csv', uvwp_exact, delimiter=',')
        np.savetxt(f'rel_error_run{r}.csv', rel_error, delimiter=',')
        np.savetxt(f'residualss_run{r}.csv', residualss, delimiter=',')
        np.savetxt(f'xt_all_run{r}.csv', xt_all, delimiter=',')

# Write and save training times
def save_training_times(r, elapsed, nrun):

    # column r: run
    training_times[:, r] = elapsed

    # Save training times
    print("\nSaving training times...\n")
    np.savetxt('training_times.csv', training_times, delimiter=',')
