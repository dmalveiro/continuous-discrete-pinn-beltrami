import deepxde as dde
import numpy as np
import torch
import prm
import numpy_to_torch

def compute_gradients(x_np, previous_model):

    if prm.gg == 0:	# Training stage
        x = numpy_to_torch.pytorch_output(x_np).requires_grad_(True)
    elif prm.gg == 1:	# Residuals prediction stage (results.py)
        x = x_np

    u_n_mat = previous_model.net(x)

    # Velocity components - n
    u_n = u_n_mat[:, 0:1]
    v_n = u_n_mat[:, 1:2]
    w_n = u_n_mat[:, 2:3]
    p_n = u_n_mat[:, 3:4]

    def derivative(aa, bb, create=True):
        return torch.autograd.grad(aa, bb, grad_outputs=torch.ones_like(aa), retain_graph=True, create_graph=create)[0]

    # Space derivatives - u_n
    u_n_x = derivative(u_n, x, create=True)[:, 0:1]
    u_n_y = derivative(u_n, x, create=True)[:, 1:2]
    u_n_z = derivative(u_n, x, create=True)[:, 2:3]
    u_n_xx = derivative(u_n_x, x, create=False)[:, 0:1]
    u_n_yy = derivative(u_n_y, x, create=False)[:, 1:2]
    u_n_zz = derivative(u_n_z, x, create=False)[:, 2:3]

    # Space derivatives - v_n
    v_n_x = derivative(v_n, x, create=True)[:, 0:1]
    v_n_y = derivative(v_n, x, create=True)[:, 1:2]
    v_n_z = derivative(v_n, x, create=True)[:, 2:3]
    v_n_xx = derivative(v_n_x, x, create=False)[:, 0:1]
    v_n_yy = derivative(v_n_y, x, create=False)[:, 1:2]
    v_n_zz = derivative(v_n_z, x, create=False)[:, 2:3]

    # Space derivatives - w_n
    w_n_x = derivative(w_n, x, create=True)[:, 0:1]
    w_n_y = derivative(w_n, x, create=True)[:, 1:2]
    w_n_z = derivative(w_n, x, create=True)[:, 2:3]
    w_n_xx = derivative(w_n_x, x, create=False)[:, 0:1]
    w_n_yy = derivative(w_n_y, x, create=False)[:, 1:2]
    w_n_zz = derivative(w_n_z, x, create=False)[:, 2:3]

    # Space derivatives - p_n
    p_n_x = derivative(p_n, x, create=False)[:, 0:1]
    p_n_y = derivative(p_n, x, create=False)[:, 1:2]
    p_n_z = derivative(p_n, x, create=False)[:, 2:3]

    u_n_der = torch.hstack((u_n_x, u_n_y, u_n_z, u_n_xx, u_n_yy, u_n_zz))
    v_n_der = torch.hstack((v_n_x, v_n_y, v_n_z, v_n_xx, v_n_yy, v_n_zz))
    w_n_der = torch.hstack((w_n_x, w_n_y, w_n_z, w_n_xx, w_n_yy, w_n_zz))
    p_n_der = torch.hstack((p_n_x, p_n_y, p_n_z))

    uN_n_mat = torch.hstack((u_n_mat, u_n_der, v_n_der, w_n_der, p_n_der))
    if prm.gg == 0: uN_n_mat = uN_n_mat.detach().cpu().numpy()

    return uN_n_mat
