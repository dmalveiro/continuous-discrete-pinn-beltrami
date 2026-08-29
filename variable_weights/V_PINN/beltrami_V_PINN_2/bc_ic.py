import deepxde as dde
import numpy as np
import pandas as pd
import os
import param
from param import x_min, x_max, t_min, t_max
from analytical_solution import u_func, v_func, w_func

spatial_domain = dde.geometry.Cuboid(xmin=x_min, xmax=x_max)
temporal_domain = dde.geometry.TimeDomain(t_min, t_max)
spatio_temporal_domain = dde.geometry.GeometryXTime(spatial_domain, temporal_domain)

# It enforces the analytical solutions as Dirichlet BCs and ICs (i.e., imposing the Beltrami flow conditions)
# at certain points in the borders (Dirichlet BC), at any time t
# at time t=0 (IC), at any point (x,y,z) of the physical domain

boundary_condition_u = dde.icbc.DirichletBC(
    spatio_temporal_domain, u_func, lambda _, on_boundary: on_boundary, component=0
)
boundary_condition_v = dde.icbc.DirichletBC(
    spatio_temporal_domain, v_func, lambda _, on_boundary: on_boundary, component=1
)
boundary_condition_w = dde.icbc.DirichletBC(
    spatio_temporal_domain, w_func, lambda _, on_boundary: on_boundary, component=2
)

initial_condition_u = dde.icbc.IC(
    spatio_temporal_domain, u_func, lambda _, on_initial: on_initial, component=0
)
initial_condition_v = dde.icbc.IC(
    spatio_temporal_domain, v_func, lambda _, on_initial: on_initial, component=1
)
initial_condition_w = dde.icbc.IC(
    spatio_temporal_domain, w_func, lambda _, on_initial: on_initial, component=2
)

def create_training_set(r):

    if param.deterministic == 1:

        file_tr = f"training_set_run{r}_ORI.csv"

        if not os.path.exists(file_tr):
            print(f"Warning: '{file_tr}' not found in the current directory. Skipping.")

        # Read the CSV files
        df_tr = pd.read_csv(file_tr, header=None)

        # Create variables
        training_set = df_tr.iloc[:,:]

    return training_set
