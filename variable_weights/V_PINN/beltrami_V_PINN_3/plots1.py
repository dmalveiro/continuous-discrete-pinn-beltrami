import numpy as np
import deepxde as dde

# Training sets, validation sets, etc
# BC = boundary condition; IC = initial condition
# See more info: https://deepxde.readthedocs.io/en/latest/_modules/deepxde/data/pde.html#PDE
def printall(data, it):

    print()
    print()
    print("num_domain:", data.num_domain)
    print("num_boundary:", data.num_boundary)
    print("num_initial:", data.num_initial)
    print("num_test:", data.num_test)
    print()
    print("train_x_all:", data.train_x_all.shape)
    print("train_x_bc", data.train_x_bc.shape)
    print("num_bcs:", data.num_bcs)
    print("train_x:", data.train_x.shape)
    print("train_aux_vars:", data.train_aux_vars)
    print("test_x:", data.test_x.shape)
    print("test_aux_vars:", data.test_aux_vars)
    print()
    print(f"DeepXDE Global Float: {dde.config.real(np)}")
    print(f"Input Data (train_x) dtype: {data.train_x.dtype}")
    print()

    np.savetxt(f'train_x_all_it{it}.csv', data.train_x_all, delimiter=',')	# Domain points + BC points + IC points
    np.savetxt(f'train_x_bc_it{it}.csv', data.train_x_bc, delimiter=',')	# BC points + IC points
    np.savetxt(f'num_bcs_it{it}.csv', data.num_bcs, delimiter=',')		# Number of BC points + IC points
    np.savetxt(f'train_x_it{it}.csv', data.train_x, delimiter=',')		# train_x = train_x_bc + train_x_all
    np.savetxt(f'test_x_it{it}.csv', data.test_x, delimiter=',')		# test_x = train_x_bc + num_test
