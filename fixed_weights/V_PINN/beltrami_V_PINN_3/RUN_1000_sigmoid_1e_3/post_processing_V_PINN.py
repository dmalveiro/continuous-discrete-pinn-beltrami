import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Get directory one level up the current one
dir_one_up = Path.cwd().parent

# Add 'dir_one_up' to the Python search
if str(dir_one_up) not in sys.path:
    sys.path.append(str(dir_one_up))

try:    # import from param
    from param import nrun, t_min, t_max, dt, n_dt, niters1_a, niters1_b
except ImportError:
    param = None

loss_file_example = "loss_0.dat"
df_example = pd.read_csv(loss_file_example, sep=r"\s+", skiprows=1)
cols = df_example.shape[1]
cols_cut = (cols - 1) // 2
cols_keep = cols - cols_cut

# Very large number
rows_min = 100000000

for i in range(nrun):

    loss_file = f"loss_{i}.dat"
    df = pd.read_csv(loss_file, sep=r"\s+", skiprows=1)

    rows_new = df.shape[0]

    if rows_new < rows_min: rows_min = rows_new


sum_losses = np.zeros((rows_min, cols_keep))

#***
total_loss_i = []
min_max1 = np.zeros((rows_min, 2))

for i in range(nrun):

    loss_file = f"loss_{i}.dat"
    df = pd.read_csv(loss_file, sep=r"\s+", skiprows=1, header=None)	# with skiprows=1, the header is ignored

    losses_i = df.iloc[:rows_min,:cols_keep].to_numpy()

    #***
    total_loss_i = np.sum(losses_i[:,1:], axis=1).reshape(-1, 1)
    if i == 0: total_losses_all = total_loss_i
    else: total_losses_all = np.hstack((total_losses_all, total_loss_i))
    for j in range(rows_min):
        min_max1[j,0] = np.min(total_losses_all[j,:])
        min_max1[j,1] = np.max(total_losses_all[j,:])

    sum_losses += losses_i

avg_losses = sum_losses / nrun


runs = avg_losses[:,0]
total_loss = np.sum(avg_losses[:,1:], axis=1)
res_loss = np.sum(avg_losses[:,1:5], axis=1)
bc_loss = np.sum(avg_losses[:,5:8], axis=1)
ic_loss = np.sum(avg_losses[:,8:11], axis=1)

avg_loss_components = np.column_stack((runs, total_loss, res_loss, bc_loss, ic_loss))

#***
total_losses_all = np.hstack((runs.reshape(-1, 1), total_losses_all))
min_max1_total_losses = np.hstack((runs.reshape(-1, 1), min_max1))

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

res_l2errors_all_runs = np.hstack((time, final_data_v2_all))
min_max2_res_l2errors = np.hstack((time, min_max2))

#########

np.savetxt('CSV_post_processing/avg_losses.csv', avg_losses, delimiter=',', fmt='%.8e')
np.savetxt('CSV_post_processing/avg_loss_components.csv', avg_loss_components, delimiter=',', fmt='%.8e')
np.savetxt('CSV_post_processing/total_loss.csv', total_loss, delimiter=',', fmt='%.8e')
np.savetxt('CSV_post_processing/res_loss.csv', res_loss, delimiter=',', fmt='%.8e')
np.savetxt('CSV_post_processing/bc_loss.csv', bc_loss, delimiter=',', fmt='%.8e')
np.savetxt('CSV_post_processing/ic_loss.csv', ic_loss, delimiter=',', fmt='%.8e')

#***
np.savetxt('CSV_post_processing/total_losses_all.csv', total_losses_all, delimiter=',', fmt='%.8e')
np.savetxt('CSV_post_processing/min_max_total_losses.csv', min_max1_total_losses, delimiter=',', fmt='%.8e')

## RESIDUALS / L2 ERRORS ########
np.savetxt('CSV_post_processing/avg_res_l2errors.csv', avg_res_l2errors, delimiter=',', fmt='%.8e')
np.savetxt('CSV_post_processing/avg_res_l2errors_v2_time.csv', avg_res_l2errors_v2_time, delimiter=',', fmt='%.8e')
np.savetxt('CSV_post_processing/res_l2errors_all_runs.csv', res_l2errors_all_runs, delimiter=',', fmt='%.8e')
np.savetxt('CSV_post_processing/min_max_res_l2errors.csv', min_max2_res_l2errors, delimiter=',', fmt='%.8e')
