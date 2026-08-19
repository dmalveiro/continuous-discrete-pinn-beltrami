import numpy as np
import deepxde as dde
import prm
from prm import time, ib

# Training sets, validation sets, etc
# BC = boundary condition; IC = initial condition
# See more info: https://deepxde.readthedocs.io/en/latest/_modules/deepxde/data/pde.html#PDE
def printall(data_ab, time_step, it):

    print()
    print()
    print("num_domain:", data_ab.num_domain)
    print("num_boundary:", data_ab.num_boundary)
    print("num_test:", data_ab.num_test)
    print()
    print("train_x_all:", data_ab.train_x_all.shape)
    print("train_x_bc", data_ab.train_x_bc.shape)
    print("num_bcs:", data_ab.num_bcs)
    print("train_x:", data_ab.train_x.shape)
    print("test_x:", data_ab.test_x.shape)
    print()
    print(f"DeepXDE Global Float: {dde.config.real(np)}")
    print(f"Input Data (train_x) dtype: {data_ab.train_x.dtype}")
    print()

    np.savetxt(f'train_x_all_it{it}_ts{time_step:04d}.csv', data_ab.train_x_all, delimiter=',')      # Domain points + BC points + IC points
    np.savetxt(f'train_x_bc_it{it}_ts{time_step:04d}.csv', data_ab.train_x_bc, delimiter=',')        # BC points + IC points
    np.savetxt(f'num_bcs_it{it}_ts{time_step:04d}.csv', data_ab.num_bcs, delimiter=',')              # Number of BC points + IC points
    np.savetxt(f'train_x_it{it}_ts{time_step:04d}.csv', data_ab.train_x, delimiter=',')              # train_x = train_x_bc + train_x_all
    np.savetxt(f'test_x_it{it}_ts{time_step:04d}.csv', data_ab.test_x, delimiter=',')                # test_x = train_x_bc + num_test
