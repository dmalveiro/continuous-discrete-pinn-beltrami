import deepxde as dde
import numpy as np
import pandas as pd
import os
import torch
import prm
from prm import x_min, x_max, t_min, t_max, n_train1, n_val, n_train2
from analytical_solution import u_func, v_func, w_func, p_func
from analytical_solution import u0_func, v0_func, w0_func, p0_func

# No time domain. Only 3D space
spatial_domain = dde.geometry.Cuboid(xmin=x_min, xmax=x_max)

# For t > 0 (Dirichlet BCs for the Loss_BC):
bc_u = dde.icbc.DirichletBC(spatial_domain, u_func, lambda _, on_boundary: on_boundary, component=0)
bc_v = dde.icbc.DirichletBC(spatial_domain, v_func, lambda _, on_boundary: on_boundary, component=1)
bc_w = dde.icbc.DirichletBC(spatial_domain, w_func, lambda _, on_boundary: on_boundary, component=2)
bc_p = dde.icbc.DirichletBC(spatial_domain, p_func, lambda _, on_boundary: on_boundary, component=3)

def create_training_set(it):

    if prm.deterministic == 1:

        file_tr = f"training_set_it{prm.it}_ORI.csv"

        if not os.path.exists(file_tr):
            print(f"Warning: '{file_tr}' not found in the current directory. Skipping.")

        # Read the CSV files
        df_tr = pd.read_csv(file_tr, header=None)

        # Create variables
        training_set = df_tr.iloc[:,:]

    return training_set
