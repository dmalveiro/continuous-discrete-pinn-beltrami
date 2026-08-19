import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Get directory one level up the current one
dir_one_up = Path.cwd().parent

# Add 'dir_one_up' to the Python search
if str(dir_one_up) not in sys.path:
    sys.path.append(str(dir_one_up))

try:    # import from prm
    from prm import niter, t_min, t_max, dt, n_dt, nepochs1_a, nepochs1_b
except ImportError:
    prm = None

## i > 0 #####
for i in range(niter):

    loss_i_csv = []

    for j in range(1, n_dt):

        loss_file = f"loss_it{i}_ts{j:04d}.dat"
        df = pd.read_csv(loss_file, sep=r"\s+", skiprows=1, header=None)

        cols = df.shape[1]
        cols_cut = (cols - 1) // 2
        cols_keep = cols - cols_cut

        epochs_local = df.iloc[:,0].to_numpy()
        loss_iti_tsj = df.iloc[:,1:cols_keep].to_numpy()
        epochs_global = epochs_local + ((nepochs1_a + 1) + (j - 1) * (nepochs1_b + 1))
        loss_iti_tsj_mod = np.hstack((epochs_global.reshape(-1, 1), loss_iti_tsj))

        if j == 1: loss_i_csv = loss_iti_tsj_mod
        else: loss_i_csv = np.vstack((loss_i_csv, loss_iti_tsj_mod))

    np.savetxt(f'loss_{i}.csv', loss_i_csv, delimiter=',', fmt='%.8e')
########

## i = 0 #####
loss_file_example_ts0 = "loss_it0_ts0000.dat"
df_example_ts0 = pd.read_csv(loss_file_example_ts0, sep=r"\s+", skiprows=1, header=None)
rows_ts0, cols_ts0 = df_example_ts0.shape
cols_cut_ts0 = (cols_ts0 - 1) // 2
cols_keep_ts0 = cols_ts0 - cols_cut_ts0
sum_losses_ts0 = np.zeros((rows_ts0, cols_keep_ts0))
#***
total_loss_i_ts0 = []
min_max1_ts0 = np.zeros((rows_ts0, 2))

## i > 0 #####
rows_loss_i, cols_loss_i = loss_i_csv.shape
sum_losses = np.zeros((rows_loss_i, cols_loss_i))
#***
total_loss_i = []
min_max1 = np.zeros((rows_loss_i, 2))

for i in range(niter):

    ## i = 0 #####
    loss_file_ts0 = f"loss_it{i}_ts0000.dat"
    df_ts0 = pd.read_csv(loss_file_ts0, sep=r"\s+", skiprows=1, header=None)
    losses_i_ts0 = df_ts0.iloc[:,:cols_keep_ts0].to_numpy()
    sum_losses_ts0 += losses_i_ts0
    #***
    total_loss_i_ts0 = np.sum(losses_i_ts0[:,1:], axis=1).reshape(-1, 1)
    if i == 0: total_losses_all_ts0 = total_loss_i_ts0
    else: total_losses_all_ts0 = np.hstack((total_losses_all_ts0, total_loss_i_ts0))
    for j in range(rows_ts0):
        min_max1_ts0[j,0] = np.min(total_losses_all_ts0[j,:])
        min_max1_ts0[j,1] = np.max(total_losses_all_ts0[j,:])

    ## i > 0 #####
    loss_file = f"loss_{i}.csv"
    df = pd.read_csv(loss_file, header=None)
    losses_i = df.iloc[:,:].to_numpy()
    sum_losses += losses_i
    #***
    total_loss_i = np.sum(losses_i[:,1:], axis=1).reshape(-1, 1)
    if i == 0: total_losses_all = total_loss_i
    else: total_losses_all = np.hstack((total_losses_all, total_loss_i))
    for j in range(rows_loss_i):
        min_max1[j,0] = np.min(total_losses_all[j,:])
        min_max1[j,1] = np.max(total_losses_all[j,:])


# i = 0
avg_losses_ts0 = sum_losses_ts0 / niter

# i > 0
avg_losses = sum_losses / niter

## i = 0 #####
iterations_ts0 = avg_losses_ts0[:,0]
total_loss_ts0 = np.sum(avg_losses_ts0[:,1:], axis=1)
avg_loss_components_ts0 = np.column_stack((iterations_ts0, total_loss_ts0))
#***
total_losses_all_ts0 = np.hstack((iterations_ts0.reshape(-1, 1), total_losses_all_ts0))
min_max1_total_losses_ts0 = np.hstack((iterations_ts0.reshape(-1, 1), min_max1_ts0))

## i > 0 #####
iterations = avg_losses[:,0]
total_loss = np.sum(avg_losses[:,1:], axis=1)
res_loss = np.sum(avg_losses[:,1:5], axis=1)
bc_loss = np.sum(avg_losses[:,5:], axis=1)
avg_loss_components = np.column_stack((iterations, total_loss, res_loss, bc_loss))
#***
total_losses_all = np.hstack((iterations.reshape(-1, 1), total_losses_all))
min_max1_total_losses = np.hstack((iterations.reshape(-1, 1), min_max1))

## RESIDUALS / L2 ERRORS ########

df_final_data = pd.read_csv('final_data.csv', header=None)
cols_final_data = df_final_data.shape[1]
iloc_final_data = df_final_data.iloc[:,:].to_numpy()
avg_res_l2errors = np.sum(iloc_final_data, axis=1) / cols_final_data

avg_res_l2errors_v2 = avg_res_l2errors.reshape(-1, 5)
time = np.arange(t_min, t_max+dt, dt)[:, np.newaxis]
avg_res_l2errors_v2_time = np.hstack((time, avg_res_l2errors_v2))

final_data_v2_all = []
min_max2 = np.zeros((n_dt, 10))     # 2x (res,u,v,w,p)

for i in range(cols_final_data):

    final_data_v2 = iloc_final_data[:,i].reshape(-1, 5)
    rows_final = final_data_v2.shape[0]

    if i == 0: final_data_v2_all = final_data_v2
    else: final_data_v2_all = np.hstack((final_data_v2_all, final_data_v2))

    for j in range(rows_final):
        for k in range(5):
            min_max2[j,2*k] = np.min(final_data_v2_all[j,k::5])
            min_max2[j,2*k+1] = np.max(final_data_v2_all[j,k::5])

res_l2errors_all_iterations = np.hstack((time, final_data_v2_all))
min_max2_res_l2errors = np.hstack((time, min_max2))

## i = 0 #####
np.savetxt('CSV_post_processing/avg_losses_ts0.csv', avg_losses_ts0, delimiter=',', fmt='%.8e')
np.savetxt('CSV_post_processing/avg_loss_components_ts0.csv', avg_loss_components_ts0, delimiter=',', fmt='%.8e')
np.savetxt('CSV_post_processing/total_loss_ts0.csv', total_loss_ts0, delimiter=',', fmt='%.8e')
#***
np.savetxt('CSV_post_processing/total_losses_all_ts0.csv', total_losses_all_ts0, delimiter=',', fmt='%.8e')
np.savetxt('CSV_post_processing/min_max_total_losses_ts0.csv', min_max1_total_losses_ts0, delimiter=',', fmt='%.8e')

## i > 0 #####
np.savetxt('CSV_post_processing/avg_losses.csv', avg_losses, delimiter=',', fmt='%.8e')
np.savetxt('CSV_post_processing/avg_loss_components.csv', avg_loss_components, delimiter=',', fmt='%.8e')
np.savetxt('CSV_post_processing/total_loss.csv', total_loss, delimiter=',', fmt='%.8e')
np.savetxt('CSV_post_processing/res_loss.csv', res_loss, delimiter=',', fmt='%.8e')
np.savetxt('CSV_post_processing/bc_loss.csv', bc_loss, delimiter=',', fmt='%.8e')
#***
np.savetxt('CSV_post_processing/total_losses_all.csv', total_losses_all, delimiter=',', fmt='%.8e')
np.savetxt('CSV_post_processing/min_max_total_losses.csv', min_max1_total_losses, delimiter=',', fmt='%.8e')

## RESIDUALS / L2 ERRORS ########
np.savetxt('CSV_post_processing/avg_res_l2errors.csv', avg_res_l2errors, delimiter=',', fmt='%.8e')
np.savetxt('CSV_post_processing/avg_res_l2errors_v2_time.csv', avg_res_l2errors_v2_time, delimiter=',', fmt='%.8e')
np.savetxt('CSV_post_processing/res_l2errors_all_iterations.csv', res_l2errors_all_iterations, delimiter=',', fmt='%.8e')
np.savetxt('CSV_post_processing/min_max_res_l2errors.csv', min_max2_res_l2errors, delimiter=',', fmt='%.8e')
