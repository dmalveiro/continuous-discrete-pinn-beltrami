import deepxde as dde
import numpy as np
import torch
import prm

# n: model at previous time instant
# n+1: model at current time instant
# def pde(inputs, outputs at t_{n+1}, outputs at t_n)

def pde(x, u_n1_mat, uN_n_mat):

    # Velocity components - n1
    u_n1 = u_n1_mat[:, 0:1]
    v_n1 = u_n1_mat[:, 1:2]
    w_n1 = u_n1_mat[:, 2:3]
    p_n1 = u_n1_mat[:, 3:4]

    # Velocity components - n
    u_n = uN_n_mat[:, 0:1]
    v_n = uN_n_mat[:, 1:2]
    w_n = uN_n_mat[:, 2:3]
    p_n = uN_n_mat[:, 3:4]

    # Space derivatives - u_n1
    u_n1_x = dde.grad.jacobian(u_n1_mat, x, i=0, j=0)
    u_n1_y = dde.grad.jacobian(u_n1_mat, x, i=0, j=1)
    u_n1_z = dde.grad.jacobian(u_n1_mat, x, i=0, j=2)
    u_n1_xx = dde.grad.hessian(u_n1_mat, x, component=0, i=0, j=0)
    u_n1_yy = dde.grad.hessian(u_n1_mat, x, component=0, i=1, j=1)
    u_n1_zz = dde.grad.hessian(u_n1_mat, x, component=0, i=2, j=2)

    # Space derivatives - v_n1
    v_n1_x = dde.grad.jacobian(u_n1_mat, x, i=1, j=0)
    v_n1_y = dde.grad.jacobian(u_n1_mat, x, i=1, j=1)
    v_n1_z = dde.grad.jacobian(u_n1_mat, x, i=1, j=2)
    v_n1_xx = dde.grad.hessian(u_n1_mat, x, component=1, i=0, j=0)
    v_n1_yy = dde.grad.hessian(u_n1_mat, x, component=1, i=1, j=1)
    v_n1_zz = dde.grad.hessian(u_n1_mat, x, component=1, i=2, j=2)

    # Space derivatives - w_n1
    w_n1_x = dde.grad.jacobian(u_n1_mat, x, i=2, j=0)
    w_n1_y = dde.grad.jacobian(u_n1_mat, x, i=2, j=1)
    w_n1_z = dde.grad.jacobian(u_n1_mat, x, i=2, j=2)
    w_n1_xx = dde.grad.hessian(u_n1_mat, x, component=2, i=0, j=0)
    w_n1_yy = dde.grad.hessian(u_n1_mat, x, component=2, i=1, j=1)
    w_n1_zz = dde.grad.hessian(u_n1_mat, x, component=2, i=2, j=2)

    # Space derivatives - p_n1
    p_n1_x = dde.grad.jacobian(u_n1_mat, x, i=3, j=0)
    p_n1_y = dde.grad.jacobian(u_n1_mat, x, i=3, j=1)
    p_n1_z = dde.grad.jacobian(u_n1_mat, x, i=3, j=2)

# ----------

    # Space derivatives - u_n
    u_n_x = uN_n_mat[:, 4:5]
    u_n_y = uN_n_mat[:, 5:6]
    u_n_z = uN_n_mat[:, 6:7]
    u_n_xx = uN_n_mat[:, 7:8]
    u_n_yy = uN_n_mat[:, 8:9]
    u_n_zz = uN_n_mat[:, 9:10]

    # Space derivatives - v_n
    v_n_x = uN_n_mat[:, 10:11]
    v_n_y = uN_n_mat[:, 11:12]
    v_n_z = uN_n_mat[:, 12:13]
    v_n_xx = uN_n_mat[:, 13:14]
    v_n_yy = uN_n_mat[:, 14:15]
    v_n_zz = uN_n_mat[:, 15:16]

    # Space derivatives - w_n
    w_n_x = uN_n_mat[:, 16:17]
    w_n_y = uN_n_mat[:, 17:18]
    w_n_z = uN_n_mat[:, 18:19]
    w_n_xx = uN_n_mat[:, 19:20]
    w_n_yy = uN_n_mat[:, 20:21]
    w_n_zz = uN_n_mat[:, 21:22]

    # Space derivatives - p_n
    p_n_x = uN_n_mat[:, 22:23]
    p_n_y = uN_n_mat[:, 23:24]
    p_n_z = uN_n_mat[:, 24:25]

# ----------

    # Time derivatives
    u_t = (u_n1 - u_n) / prm.dt
    v_t = (v_n1 - v_n) / prm.dt
    w_t = (w_n1 - w_n) / prm.dt

    # Midpoint velocities and pressure
    u_mid = 0.5 * (u_n1 + u_n)
    v_mid = 0.5 * (v_n1 + v_n)
    w_mid = 0.5 * (w_n1 + w_n)
    p_mid = 0.5 * (p_n1 + p_n)

    # Midpoint derivatives - u
    u_mid_x = 0.5 * (u_n1_x + u_n_x)
    u_mid_y = 0.5 * (u_n1_y + u_n_y)
    u_mid_z = 0.5 * (u_n1_z + u_n_z)
    u_mid_xx = 0.5 * (u_n1_xx + u_n_xx)
    u_mid_yy = 0.5 * (u_n1_yy + u_n_yy)
    u_mid_zz = 0.5 * (u_n1_zz + u_n_zz)

    # Midpoint derivatives - v
    v_mid_x = 0.5 * (v_n1_x + v_n_x)
    v_mid_y = 0.5 * (v_n1_y + v_n_y)
    v_mid_z = 0.5 * (v_n1_z + v_n_z)
    v_mid_xx = 0.5 * (v_n1_xx + v_n_xx)
    v_mid_yy = 0.5 * (v_n1_yy + v_n_yy)
    v_mid_zz = 0.5 * (v_n1_zz + v_n_zz)

    # Midpoint derivatives - w
    w_mid_x = 0.5 * (w_n1_x + w_n_x)
    w_mid_y = 0.5 * (w_n1_y + w_n_y)
    w_mid_z = 0.5 * (w_n1_z + w_n_z)
    w_mid_xx = 0.5 * (w_n1_xx + w_n_xx)
    w_mid_yy = 0.5 * (w_n1_yy + w_n_yy)
    w_mid_zz = 0.5 * (w_n1_zz + w_n_zz)

    # Midpoint derivatives - p
    p_mid_x = 0.5 * (p_n1_x + p_n_x)
    p_mid_y = 0.5 * (p_n1_y + p_n_y)
    p_mid_z = 0.5 * (p_n1_z + p_n_z)

    N_mid_mx = (
        (u_mid * u_mid_x + v_mid * u_mid_y + w_mid * u_mid_z)
        + p_mid_x
        - 1 / prm.Re * (u_mid_xx + u_mid_yy + u_mid_zz)
    )

    N_mid_my = (
        (u_mid * v_mid_x + v_mid * v_mid_y + w_mid * v_mid_z)
        + p_mid_y
        - 1 / prm.Re * (v_mid_xx + v_mid_yy + v_mid_zz)
    )

    N_mid_mz = (
        (u_mid * w_mid_x + v_mid * w_mid_y + w_mid * w_mid_z)
        + p_mid_z
        - 1 / prm.Re * (w_mid_xx + w_mid_yy + w_mid_zz)
    )

    momentum_x = u_t + N_mid_mx
    momentum_y = v_t + N_mid_my
    momentum_z = w_t + N_mid_mz
    continuity = u_mid_x + v_mid_y + w_mid_z

    return [momentum_x, momentum_y, momentum_z, continuity]
